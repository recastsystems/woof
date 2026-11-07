// WRF v4.6.1 Goddard 4-ice microphysics with hail (NU-WRF), mp_physics = 7.
//
// Source: phys/module_mp_gsfcgce_4ice_nuwrf.F at WRF d66e442 (4,995
// lines, public domain).  Transcribed in WRF's operation order:
//   consat_s        -> gsfc4ice_consat (one thread, once per device, into
//                      the constant table every column launch reads)
//   fall_flux       -> g4_fall_flux (one column per thread)
//   saticel_s       -> g4_saticel   (the same thread, level by level)
//   sgmap, vqrqi, gammagce, gamma_toshi, eff_rad -> device functions
// negcor exists in the module but its only callers are commented out
// (:410-433); auto_conversion and the GOCART branches compile only under
// WRF_CHEM; refl10cm_gsfc is defined and never called.  None of the four
// is reachable from a WRF 4.6.1 mp_physics=7 run, so none is here.  The
// scheme's REFL_10CM is the DBZ saticel_s computes itself (:4276-4318),
// written when the diagnostic is due (:4493-4503).
//
// ARITHMETIC.  The referee is the scheme object WRF's own build produced
// (configure 32, -O2 -fno-fast-math -ffp-contract=off), linked unchanged by
// tools/gsfcgce4ice_wrf461_oracle.  What that object computes, and so what
// this file computes:
//   * every REAL**REAL is powf (strict g4_pow -> host-libm gfk_pow). Only the
//     real exponents 2.0, -1.0 and 1.0 fold, and the scheme writes none;
//   * an INTEGER power is GCC's powi_table multiply chain:
//       x**2 = x*x, x**3 = x*(x*x), x**5 = (x*x)*(x*(x*x));
//     (a -O0 build calls libgcc __powisf2 instead and rounds x**5
//     differently -- the oracle records that split, it is not WRF's);
//   * strict EXP uses gfk_exp; strict ALOG10 uses CORE-MATH for glibc
//     2.41+ and the legacy gfk_log decomposition on older referees;
//     gammagce's double LOG uses the pinned binary64 log;
//   * an expression of literals only is folded by gfortran at compile time
//     with correctly rounded arithmetic: 4.*atan(1.), exp(0.304*40.0) and
//     4.464**0.95 enter as the folded words;
//   * MAX(a,b) returns b on a tie or NaN, MIN(a,b) returns a -- the maxss /
//     minss operand order of the referee's own instructions;
//   * Fortran evaluates a*b*c/d left to right; every product and quotient
//     below keeps that grouping, and every division is __fdiv_rn so NVRTC
//     cannot rewrite one by a constant.
// Under GPUWM_WRF_EXACT=1 the loader compiles with --fmad=false and the
// result is the referee's word. The default uses CUDA powf/expf/log10f and
// contraction; its residual is measured, not assumed.
//
// Units are WRF's: the scheme converts to cgs internally (:2439-2456) and
// so does this transcription.  Array layout is gpuwm's (k, column), column
// fastest; one thread owns one column for the whole call.

#ifndef G4_TABLE_WORDS
#define G4_TABLE_WORDS 512
#endif

#define G4DIV(a, b) __fdiv_rn((a), (b))
#define G4SQRT(a) __fsqrt_rn(a)

__device__ __forceinline__ float g4_max(float a, float b) { return (a > b) ? a : b; }
__device__ __forceinline__ float g4_min(float a, float b) { return (b < a) ? b : a; }
// Fast production math; host-libm emulation is a proof-only opt-in.
#ifdef GPUWM_WRF_EXACT
__device__ __forceinline__ float g4_pow(float x, float y) { return gfk_pow(x, y); }
__device__ __forceinline__ float g4_exp(float x) { return gfk_exp(x); }
#else
__device__ __forceinline__ float g4_pow(float x, float y) { return powf(x, y); }
__device__ __forceinline__ float g4_exp(float x) { return expf(x); }
#endif
// x**2, x**3, x**5 as WRF's build expands them (powi_table chain).
__device__ __forceinline__ float g4_p2(float x) { return x * x; }
__device__ __forceinline__ float g4_p3(float x) { return x * (x * x); }
__device__ __forceinline__ float g4_p5(float x) { float x2 = x * x; return x2 * (x * x2); }

__device__ __forceinline__ float g4_bits(unsigned int u) { return __uint_as_float(u); }

// ALOG10 on WOOF's own logf.  The decomposition (exponent split, then
// y*log10(2)_lo + log10(e)*logf(m) + y*log10(2)_hi) can be measured against the oracle host's
// log10f by tools/gsfcgce4ice_wrf461_oracle/log10_sweep.c. The numerical
// column grade remains the admission check.
__device__ float g4_log10(float x)
{
#ifndef GPUWM_WRF_EXACT
    return log10f(x);
#elif defined(GPUWM_G4_CORE_LOG10)
    return g4_cr_log10f(x);
#else
    const float two25 = g4_bits(0x4c000000u);
    const float ivln10 = g4_bits(0x3ede5bd9u);
    const float log10_2hi = g4_bits(0x3e9a2080u);
    const float log10_2lo = g4_bits(0x355427dbu);
    int hx = __float_as_int(x);
    int k = 0;
    if (hx < 0x00800000) {
        if ((hx & 0x7fffffff) == 0) return G4DIV(-two25, fabsf(x));
        if (hx < 0) return G4DIV(x - x, x - x);
        k -= 25;
        x = x * two25;
        hx = __float_as_int(x);
    }
    if (hx >= 0x7f800000) return x + x;
    k += (hx >> 23) - 127;
    int i = (int)(((unsigned int)k & 0x80000000u) >> 31);
    hx = (hx & 0x007fffff) | ((0x7f - i) << 23);
    float y = (float)(k + i);
    x = __int_as_float(hx);
    float z = y * log10_2lo + ivln10 * gfk_log(x);
    return z + y * log10_2hi;
#endif
}

// ---------------------------------------------------------------------------
// The constant table.  consat_s's module variables plus the per-call
// constants saticel_s derives from them without the time step.
// ---------------------------------------------------------------------------
struct G4K {
    float cmin, al, cp, rd1, rd2, cpi, cpi2;
    float c358, c610, c149, c879, c172, c409, c76, c218, c580, c141;
    float t0, t00, alv, alf, als, avc, afc, asc;
    float tnh, roqh, ah, bh, roqg, ag, bg, ag2, bg2, tng, roqg2, qrog2;
    float tns, roqs, as, bs, aw, bw, roqr, tnw;
    float bgh, bgh2, bsh, bwh, bhh, bgq, bgq2, bsq, bwq, bhq;
    float tslopes, tslopeg, draimax, dsnomin, dsnomin4, dgrpmin, dgrpmin4;
    float xs, sno11, sno00, dsno11, dsno00, sexp11, sexp00, stt, stexp, slim, sbase;
    float xg, grp11, grp00, dgrp11, dgrp00, gexp11, gexp00, gtt, gtexp, glim, gbase;
    float hai00, hai11, htt0, htt1, haixp;
    float zrc, zsc, zgc, zgc2, zhc, vrc, vrc0, vrc1, vrc2, vrc3, vsc, vgc, vgc2, vhc;
    float rn3, rn4, rn5, rn50, rn51, rn52, rn53, rn6, rn60, rn61, rn62, rn63;
    float rn7, rn8, rn9, rn10, rn101, rn102, rn10a, rn10b, rn10c, rn11, rn11a;
    float ami50, ami100, ami40, rn12;
    float rn14, rn142, rn15, rn15a, rn15a2, rn16, rn17, rn17a2, rn17b, rn17c, rn17a;
    float rn18, rn18a, rn191, rn192, rn192_2, rn19, rn19a;
    float rn20, rn20a, rn20b, rn30a, rn30, rn21, bnd21, rn22, rn23, rn231, rn232;
    float rn25, rn10c_dup, rn32, rn33, rn331, rn332, rn34;
    float hn9, hn10, hn14, hn15a, hn16, hn17, hn17a, hn19, hn19a, hn10a, hn20, hn20b;
    float rn12a[31], rn12b[31], rn13[31], aa1[31], aa2[31];
    // saticel_s per-call constants independent of dt (:4281-4286), and the
    // cloud-radius gamma ratios for the two XLAND values (:4412-4428).
    float ucor, ucos, ucog, ucog2, ucoh, uwet;
    float gamfac1_land, gamfac3_land, gamfac1_water, gamfac3_water;
};

static_assert(sizeof(G4K) <= 4 * G4_TABLE_WORDS, "G4K outgrew its table");

// gammagce (:1836-1862): real*8 Lanczos with DEFAULT-REAL data literals,
// so each coefficient is a float widened to double.
__device__ const double G4_GAMMAGCE_COF[6] = {
    (double)76.18009173f, (double)-86.50532033f, (double)24.01409822f,
    (double)-1.231739516f, (double).120858003e-2f, (double)-.536382e-5f};

__device__ float g4_gammagce(float xx)
{
    const double *cof = G4_GAMMAGCE_COF;
    const double stp = (double)2.50662827465f;
    const double half = 0.5, one = 1.0, fpf = 5.5;
    double x = (double)xx - one;
    double tmp = x + fpf;
    tmp = __dsub_rn(__dmul_rn(x + half, glibc_log(tmp)), tmp);
    double ser = one;
    for (int j = 0; j < 6; ++j) {
        x = x + one;
        ser = __dadd_rn(ser, __ddiv_rn(cof[j], x));
    }
    float gammln = __double2float_rn(__dadd_rn(tmp, glibc_log(__dmul_rn(stp, ser))));
    return g4_exp(gammln);
}

// gamma_toshi (:4682-4747) for the positive non-integer arguments the
// cloud radius gives it (mu+1, mu+3, mu+4 with mu = 8/3 or 26/3).
__device__ const double G4_GAMMA_TOSHI_G[26] = {
        1.0, 0.5772156649015329, -0.6558780715202538, -0.420026350340952e-1,
        0.1665386113822915, -.421977345555443e-1, -.96219715278770e-2,
        .72189432466630e-2, -.11651675918591e-2, -.2152416741149e-3,
        .1280502823882e-3, -.201348547807e-4, -.12504934821e-5,
        .11330272320e-5, -.2056338417e-6, .61160950e-8, .50020075e-8,
        -.11812746e-8, .1043427e-9, .77823e-11, -.36968e-11, .51e-12,
        -.206e-13, -.54e-14, .14e-14, .1e-15};

__device__ float g4_gamma_toshi(float x)
{
    const double *g = G4_GAMMA_TOSHI_G;
    double z = fabs((double)x);
    int m = (int)z;
    double r = 1.0;
    for (int k = 1; k <= m; ++k) r = __dmul_rn(r, z - (double)k);
    z = z - (double)m;
    double gr = g[25];
    for (int k = 24; k >= 0; --k) gr = __dadd_rn(__dmul_rn(gr, z), g[k]);
    double ga = __ddiv_rn(1.0, __dmul_rn(gr, z));
    ga = __dmul_rn(ga, r);
    return __double2float_rn(ga);
}

// consat_s's two DATA tables (:130-144).
__device__ const float G4_AA1[31] = {
    .7939e-7f, .7841e-6f, .3369e-5f, .4336e-5f, .5285e-5f,
    .3728e-5f, .1852e-5f, .2991e-6f, .4248e-6f, .7434e-6f,
    .1812e-5f, .4394e-5f, .9145e-5f, .1725e-4f, .3348e-4f,
    .1725e-4f, .9175e-5f, .4412e-5f, .2252e-5f, .9115e-6f,
    .4876e-6f, .3473e-6f, .4758e-6f, .6306e-6f, .8573e-6f,
    .7868e-6f, .7192e-6f, .6513e-6f, .5956e-6f, .5333e-6f,
    .4834e-6f};
__device__ const float G4_AA2[31] = {
    .4006f, .4831f, .5320f, .5307f, .5319f, .5249f, .4888f, .3894f,
    .4047f, .4318f, .4771f, .5183f, .5463f, .5651f, .5813f, .5655f,
    .5478f, .5203f, .4906f, .4447f, .4126f, .3960f, .4149f, .4320f,
    .4506f, .4483f, .4460f, .4433f, .4413f, .4382f, .4361f};

// consat_s (:1347-1826), in its own statement order, written straight into
// the table (no per-thread copy of the struct, so the unit carries no
// local frame).
extern "C" __global__ void gsfc4ice_consat(float *table)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int i = 0; i < G4_TABLE_WORDS; ++i) table[i] = 0.0f;
    G4K &c = *reinterpret_cast<G4K *>(table);
    const float *aa1 = G4_AA1;
    const float *aa2 = G4_AA2;

    c.cmin = 1.e-20f;
    c.al = 2.5e10f;
    c.cp = 1.004e7f;
    c.rd1 = 1.e-3f;
    c.rd2 = 2.2f;
    // cpi = 4.*atan(1.): folded by gfortran, correctly rounded atan(1.).
    c.cpi = g4_bits(0x40490fdbu);
    c.cpi2 = c.cpi * c.cpi;
    const float grvt = 980.f;
    c.c358 = 35.86f;
    c.c610 = 6.1078e3f;
    c.c149 = 1.496286e-5f;
    c.c879 = 8.794142f;
    c.c172 = 17.26939f;
    c.c409 = 4098.026f;
    c.c76 = 7.66f;
    c.c218 = 21.87456f;
    c.c580 = 5807.695f;
    c.c141 = 1.414435e7f;
    const float tca = 2.43e3f, dwv = .226f, dva = 1.718e-4f, amw = 18.016f,
                ars = 8.314e7f;
    c.t0 = 273.16f;
    c.t00 = 238.16f;
    c.alv = 2.5e10f;
    c.alf = 3.336e9f;
    c.als = 2.8336e10f;
    c.avc = G4DIV(c.alv, c.cp);
    c.afc = G4DIV(c.alf, c.cp);
    c.asc = G4DIV(c.als, c.cp);
    const float rw = 4.615e6f, cw = 4.187e7f, ci = 2.093e7f;
    // hail
    c.tnh = 0.01f;
    c.roqh = 0.9f;
    const float cd1 = 6.e-1f;
    const float cd2 = G4DIV(4.f * grvt, 3.f * cd1);
    c.ah = G4SQRT(cd2 * c.roqh);
    c.bh = 0.5f;
    // graupel
    c.roqg = .3f;
    c.ag = 330.22f;
    c.bg = .36f;
    c.ag2 = 544.83f;
    c.bg2 = .54f;
    c.tng = .04f;
    c.roqg2 = 0.5f;
    c.qrog2 = 2.0f;
    c.qrog2 = c.qrog2 * 1.e-6f;
    // snow (the second assignments win, :1507-1510)
    c.tns = 0.1f;
    c.roqs = .05f;
    c.as = 151.01f;
    c.bs = 0.24f;
    // rain
    c.aw = 2115.f;
    c.bw = .8f;
    c.roqr = 1.f;
    c.tnw = .08f;
    c.bgh = .5f * c.bg;
    c.bgh2 = .5f * c.bg2;
    c.bsh = .5f * c.bs;
    c.bwh = .5f * c.bw;
    c.bhh = .5f * c.bh;
    c.bgq = .25f * c.bg;
    c.bgq2 = .25f * c.bg2;
    c.bsq = .25f * c.bs;
    c.bwq = .25f * c.bw;
    c.bhq = .25f * c.bh;
    c.tslopes = 0.11177209f;
    c.tslopeg = 0.05756866f;
    c.draimax = 0.0500f;
    c.draimax = g4_pow(c.draimax, 4.f) * c.roqr * c.cpi;
    c.dsnomin = 0.0100f;
    c.dsnomin4 = g4_pow(c.dsnomin, 4.f) * 0.900f * c.cpi;
    c.dgrpmin = 0.0135f;
    c.dgrpmin4 = g4_pow(c.dgrpmin, 4.f) * c.roqg * c.cpi;
    c.xs = 0.97f; c.sno11 = 0.70f; c.sno00 = -0.24f; c.dsno11 = 3.25f;
    c.dsno00 = 0.40f; c.sexp11 = 1.5f; c.sexp00 = 0.9f; c.stt = -35.f;
    c.stexp = 0.42f; c.slim = 1.00f; c.sbase = 0.04000f;
    c.xg = 0.98f; c.grp11 = 0.30f; c.grp00 = 0.30f; c.dgrp11 = 2.70f;
    c.dgrp00 = 2.60f; c.gexp11 = 0.20f; c.gexp00 = 0.20f; c.gtt = -35.f;
    c.gtexp = 0.40f; c.glim = 0.90f; c.gbase = 0.0130f;
    c.hai00 = 3.25f; c.hai11 = 0.50f; c.htt0 = -5.00f; c.htt1 = -50.00f;
    c.haixp = 2.7f;
    // gamma functions (:1572-1620).  ga3g/ga4g/ga5gh and ga3d/ga4d/ga5dh are
    // computed by gammagce and then replaced by literals selected by bg and
    // bs, so only the bw and hail values reach a result.
    const float ga3 = 2.f, ga4 = 6.f, ga5 = 24.f, ga6 = 120.f, ga7 = 720.f,
                ga8 = 5040.f, ga9 = 40320.f;
    const float ga4b = g4_gammagce(4.f + c.bw);
    const float ga5bh = g4_gammagce(G4DIV(5.f + c.bw, 2.f));
    float ga4g = 11.63177f, ga3g = 3.3233625f, ga5gh = 1.608355f;
    if (c.bg == 0.37f) { ga4g = 9.730877f; ga3g = 2.887512f; ga5gh = 1.526425f; }
    if (c.bg == 0.36f) { ga4g = 9.599978f; ga3g = 2.857136f; ga5gh = 1.520402f; }
    float ga4g2 = 0.f, ga3g2 = 0.f, ga5gh2 = 0.f;
    if (c.bg2 == 0.54f) { ga4g2 = 12.298653f; ga3g2 = 3.474196f; ga5gh2 = 1.635061f; }
    float ga3d = 2.54925f, ga4d = 8.285063f, ga5dh = 1.456943f;
    if (c.bs == 0.57f) { ga3d = 3.59304f; ga4d = 12.82715f; ga5dh = 1.655588f; }
    if (c.bs == 0.24f) { ga3d = 2.523508f; ga4d = 8.176166f; ga5dh = 1.451396f; }
    if (c.bs == 0.11f) { ga3d = 2.218906f; ga4d = 6.900796f; ga5dh = 1.382792f; }
    float ga6d = 144.93124f;
    if (c.bs == 0.24f) ga6d = 181.654791f;
    const float ga3h = g4_gammagce(3.f + c.bh);
    const float ga4h = g4_gammagce(4.f + c.bh);
    const float ga5hh = g4_gammagce(G4DIV(5.f + c.bh, 2.f));
    const float ac1 = c.aw;
    c.zrc = g4_pow(c.cpi * c.roqr * c.tnw, 0.25f);
    c.zsc = g4_pow(c.cpi * c.roqs * c.tns, 0.25f);
    c.zgc = g4_pow(c.cpi * c.roqg * c.tng, 0.25f);
    c.zgc2 = g4_pow(c.cpi * c.roqg2 * c.tng, 0.25f);
    c.zhc = g4_pow(c.cpi * c.roqh * c.tnh, 0.25f);
    c.vrc = G4DIV(c.aw * ga4b, 6.f * g4_pow(c.zrc, c.bw));
    c.vrc0 = -26.7f;
    c.vrc1 = G4DIV(20600.f, c.zrc);
    c.vrc2 = G4DIV(-204500.f, c.zrc * c.zrc);
    c.vrc3 = G4DIV(906000.f, c.zrc * c.zrc * c.zrc);
    c.vsc = G4DIV(c.as * ga4d, 6.f * g4_pow(c.zsc, c.bs));
    c.vgc = G4DIV(c.ag * ga4g, 6.f * g4_pow(c.zgc, c.bg));
    c.vgc2 = G4DIV(c.ag2 * ga4g2, 6.f * g4_pow(c.zgc2, c.bg2));
    c.vhc = G4DIV(c.ah * ga4h, 6.f * g4_pow(c.zhc, c.bh));
    const float esi = 0.70f;
    c.rn3 = .25f * c.cpi * c.tns * c.as * esi * ga3d;
    const float esc = 0.45f;
    c.rn4 = .25f * c.cpi * esc * c.tns * c.as * ga3d;
    const float eri = .1f;
    c.rn5 = .25f * c.cpi * eri * c.tnw;
    c.rn50 = -(.267e2f * ga3);
    c.rn51 = 5.15e3f * ga4;
    c.rn52 = -(1.0225e4f * ga5);
    c.rn53 = 7.55e3f * ga6;
    const float ami = G4DIV(1.f, 24.f * 6.e-9f);
    c.rn6 = c.cpi2 * eri * c.tnw * c.roqr * ami;
    c.rn60 = -(.267e2f * ga6);
    c.rn61 = 5.15e3f * ga7;
    c.rn62 = -(1.0225e4f * ga8);
    c.rn63 = 7.55e3f * ga9;
    const float esr = 1.f;
    c.rn7 = c.cpi2 * esr * c.tnw * c.tns * c.roqs;
    c.rn8 = c.cpi2 * esr * c.tnw * c.tns * c.roqr;
    const float egs = .1f;
    c.rn9 = c.cpi2 * egs * c.tns * c.tng * c.roqs;
    c.rn10 = 4.f * c.tns;
    c.rn101 = .65f;
    c.rn102 = .44f * G4SQRT(G4DIV(c.as, dva)) * ga5dh;
    c.rn10a = G4DIV(c.alv * c.als * amw, tca * ars);
    c.rn10b = G4DIV(c.alv, tca);
    c.rn10c = G4DIV(ars, dwv * amw);
    c.rn11 = G4DIV(2.f * c.cpi * c.tns * tca, c.alf);
    c.rn11a = G4DIV(cw, c.alf);
    c.ami50 = 4.8e-7f * g4_p3(G4DIV(G4DIV(c.dsnomin * 1.e4f, 2.f), 50.f));
    c.ami100 = 1.51e-7f;
    c.ami40 = 2.46e-7f * 0.669921875f;   // .875**3, folded exactly
    const float eiw = 1.f, ui50 = 100.f;
    const float ri50 = G4DIV(c.dsnomin, 2.f);
    const float cmn = 1.05e-15f;
    c.rn12 = c.cpi * eiw * ui50 * ri50 * ri50;
    for (int k = 0; k < 31; ++k) {
        c.aa1[k] = aa1[k];
        c.aa2[k] = aa2[k];
        const float y1 = 1.f - aa2[k];
        c.rn13[k] = G4DIV(aa1[k] * y1, g4_pow(c.ami50, y1) - g4_pow(c.ami40, y1));
        c.rn12a[k] = G4DIV(c.rn13[k], c.ami50);
        c.rn12b[k] = aa1[k] * g4_pow(c.ami50, aa2[k]);
        // rn25a and BergCon1..4 are written to module storage and read by
        // nothing in the 4.6.1 scheme; their pow calls are not carried.
        (void)cmn;
    }
    const float egc = 0.65f;
    c.rn14 = .25f * c.cpi * egc * c.ag * c.tng * ga3g;
    c.rn142 = .25f * c.cpi * egc * c.ag2 * c.tng * ga3g2;
    const float egi = .1f;
    c.rn15 = .25f * c.cpi * egi * c.tng * c.ag * ga3g;
    c.rn15a = .25f * c.cpi * egi * c.tng * c.ag * ga3g;
    c.rn15a2 = .25f * c.cpi * egi * c.tng * c.ag2 * ga3g2;
    const float egr = 1.f;
    c.rn16 = c.cpi2 * egr * c.tng * c.tnw * c.roqr;
    c.rn17 = 2.f * c.cpi * c.tng;
    c.rn17a2 = .31f * ga5gh2 * G4SQRT(G4DIV(c.ag2, dva));
    c.rn17b = cw - ci;
    c.rn17c = cw;
    c.rn17a = .31f * ga5gh * G4SQRT(G4DIV(c.ag, dva));
    const float apri = .66f;
    float bpri = 1.e-4f;
    bpri = 0.5f * bpri;
    c.rn18 = 20.f * c.cpi2 * bpri * c.tnw * c.roqr;
    c.rn18a = apri;
    c.rn191 = .78f;
    c.rn192 = .31f * ga5gh * G4SQRT(G4DIV(c.ag, dva));
    c.rn192_2 = .31f * ga5gh2 * G4SQRT(G4DIV(c.ag2, dva));
    c.rn19 = G4DIV(2.f * c.cpi * c.tng * tca, c.alf);
    c.rn19a = G4DIV(cw, c.alf);
    c.rn20 = 2.f * c.cpi * c.tng;
    c.rn20a = G4DIV(c.als * c.als * amw, tca * ars);
    c.rn20b = G4DIV(c.als, tca);
    c.rn30a = G4DIV(c.alv * c.alv * amw, tca * ars);
    c.rn30 = 2.f * c.cpi * c.tng;
    c.rn21 = 1.e-3f;
    c.bnd21 = 1.5e-3f;
    const float erc = 1.f;
    c.rn22 = .25f * c.cpi * erc * c.tnw;
    c.rn23 = 2.f * c.cpi * c.tnw;
    c.rn231 = .78f;
    c.rn232 = .31f * ga3 * G4SQRT(G4DIV(3.e3f, dva));
    (void)ga5bh; (void)ac1;
    const float cn0 = 1.e-8f;
    c.rn25 = cn0;
    c.rn10c_dup = G4DIV(ars, dwv * amw);
    c.rn32 = 4.f * 51.545e-4f;
    c.rn33 = 4.f * c.tns;
    c.rn331 = .65f;
    c.rn332 = .44f * G4SQRT(G4DIV(c.as, dva)) * ga5dh;
    const float amc = G4DIV(1.f, 24.f * 4.e-9f);
    c.rn34 = c.cpi2 * esc * amc * c.as * c.roqs * c.tns * ga6d;
    const float ehs = 1.f;
    c.hn9 = c.cpi2 * ehs * c.tns * c.tnh * c.roqs;
    const float ehg = 0.3f;
    c.hn10 = c.cpi2 * ehg * c.tng * c.tnh * c.roqg;
    const float ehw = 1.f;
    c.hn14 = .25f * c.cpi * ehw * c.tnh * ga3h * c.ah;
    const float ehi = 1.f;
    c.hn15a = .25f * c.cpi * ehi * c.tnh * ga3h * c.ah;
    const float ehr = 1.f;
    c.hn16 = c.cpi2 * ehr * c.tnh * c.tnw * c.roqr;
    c.hn17 = 2.f * c.cpi * c.tnh;
    c.hn17a = .31f * ga5hh * G4SQRT(c.ah);
    c.hn19 = G4DIV(2.f * c.cpi * c.tnh, c.alf);
    c.hn19a = .31f * ga5hh * G4SQRT(c.ah);
    c.hn10a = G4DIV(c.als * c.als, rw);
    c.hn20 = 2.f * c.cpi * c.tnh;
    c.hn20b = .31f * ga5hh * G4SQRT(c.ah);
    // JDC Schmidt-number term (:1817-1824)
    const float sc13 = 0.8420526f;
    c.rn102 = c.rn102 * sc13;
    c.rn17a = c.rn17a * sc13;
    c.rn17a2 = c.rn17a2 * sc13;
    c.rn192 = c.rn192 * sc13;
    c.rn192_2 = c.rn192_2 * sc13;
    c.rn232 = c.rn232 * sc13;
    c.rn332 = c.rn332 * sc13;

    // saticel_s :4281-4286, recomputed by WRF at every level with the same
    // operands; computing them once here is the same word.
    c.ucor = G4DIV(3071.29f, g4_pow(c.tnw, 0.75f));
    c.ucos = G4DIV(687.97f * g4_pow(c.roqs, 0.25f), g4_pow(c.tns, 0.75f));
    c.ucog = G4DIV(687.97f * g4_pow(c.roqg, 0.25f), g4_pow(c.tng, 0.75f));
    c.ucog2 = G4DIV(687.97f * g4_pow(c.roqg2, .25f), g4_pow(c.tng, .75f));
    c.ucoh = G4DIV(687.97f * g4_pow(c.roqh, 0.25f), g4_pow(c.tnh, 0.75f));
    c.uwet = g4_bits(0x40848d72u);   // 4.464**0.95, folded by gfortran
    // Cloud radius (:4423-4425) for XLAND = 1 (ccn 1500) and 2 (ccn 150).
    {
        const float mu = g4_min(15.e0f, G4DIV(1000.E0f, 1500.f) + 2.e0f);
        const float g4p = g4_gamma_toshi(mu + 4.e0f);
        c.gamfac3_land = G4DIV(g4p, g4_gamma_toshi(mu + 3.e0f));
        c.gamfac1_land = G4DIV(g4_gamma_toshi(mu + 4.e0f), g4_gamma_toshi(mu + 1.e0f));
    }
    {
        const float mu = g4_min(15.e0f, G4DIV(1000.E0f, 150.f) + 2.e0f);
        const float g4p = g4_gamma_toshi(mu + 4.e0f);
        c.gamfac3_water = G4DIV(g4p, g4_gamma_toshi(mu + 3.e0f));
        c.gamfac1_water = G4DIV(g4_gamma_toshi(mu + 4.e0f), g4_gamma_toshi(mu + 1.e0f));
    }
}

// ---------------------------------------------------------------------------
// sgmap (:1865-1993)
// ---------------------------------------------------------------------------
__device__ float g4_sgmap(const G4K &c, int isg, float qcs, float qcg,
                          float qcg2, float qch, float qch2, float r00,
                          float tairc)
{
    float ftnsg = 1.f;
    float qsg = 0.f;
    if (isg == 1 || isg == 3) qsg = qcs;
    if (isg == 2) qsg = qcg;
    if (isg == 4) qsg = qch;
    if (qsg > c.cmin) {
        const float qsg1 = qsg * r00 * 1.e6f;
        if (isg == 1 || isg == 3) {
            const float taird = g4_min(0.f, g4_max(c.stt, tairc) + 0.0f);
            float ftnsT = g4_exp(-c.tslopes * taird);
            float sno1 = c.sno11, dsno1 = c.dsno11, sexp1 = c.sexp11;
            if (taird > c.stt) {
                const float r = G4DIV(taird, c.stt);
                sno1 = c.sno00 - (c.sno00 - c.sno11) * g4_pow(r, c.stexp);
                dsno1 = c.dsno00 - (c.dsno00 - c.dsno11) * g4_pow(r, c.stexp);
                sexp1 = c.sexp00 - (c.sexp00 - c.sexp11) * g4_pow(r, c.stexp);
            }
            float hx = 0.f, gx = 0.f, hgx = 1.f;
            if (qch2 * 1.e6f * r00 > 0.008f)
                hx = g4_max(1.f, qch2 * 1.e6f * r00 * 125.f);
            if (qcg2 * 1.e6f * r00 > 0.040f)
                gx = g4_max(1.f, qcg2 * 1.e6f * r00 * 25.f);
            hgx = hx + gx;
            hgx = g4_max(1.f, hgx);
            const float xx = c.xs - c.xs * g4_min(c.slim,
                g4_pow(g4_max(0.f, G4DIV(qsg1 - sno1, dsno1)), sexp1));
            ftnsT = g4_pow(ftnsT, xx);
            const float fexp = xx;
            const float ftnsQ = g4_pow(G4DIV(qsg1, c.sbase), fexp);
            ftnsg = ftnsT * ftnsQ * hgx;
            const float tnsmax = G4DIV(r00 * qsg, c.dsnomin4);
            if (ftnsg * c.tns > tnsmax) ftnsg = G4DIV(tnsmax, c.tns);
            if (isg == 3) {
                const float densno = 0.001996f * g4_pow(
                    G4DIV(G4DIV(ftnsg * c.tns, qsg), r00), 0.2995f);
                ftnsg = G4DIV(g4_min(densno, 0.9f), c.roqs);
            }
        } else if (isg == 2) {
            const float taird = g4_min(0.f, g4_max(c.gtt, tairc) + 0.0f);
            float ftngT = g4_exp(-c.tslopeg * taird);
            float grp1 = c.grp11, dgrp1 = c.dgrp11, gexp1 = c.gexp11;
            if (taird > c.gtt) {
                const float r = G4DIV(taird, c.gtt);
                grp1 = c.grp00 - (c.grp00 - c.grp11) * g4_pow(r, c.gtexp);
                dgrp1 = c.dgrp00 - (c.dgrp00 - c.dgrp11) * g4_pow(r, c.gtexp);
                gexp1 = c.gexp00 - (c.gexp00 - c.gexp11) * g4_pow(r, c.gtexp);
            }
            const float xx = c.xg - c.xg * g4_min(c.glim,
                g4_pow(g4_max(0.0f, G4DIV(qsg1 - grp1, dgrp1)), gexp1));
            ftngT = g4_pow(ftngT, xx);
            const float fexp = xx;
            const float ftngQ = g4_pow(G4DIV(qsg1, c.gbase), fexp);
            ftnsg = ftngT * ftngQ;
            const float tngmax = G4DIV(r00 * qsg, c.dgrpmin4);
            if (ftnsg * c.tng > tngmax) ftnsg = G4DIV(tngmax, c.tng);
        } else if (isg == 4) {
            float hai2 = c.hai00;
            if (tairc <= c.htt0 && tairc >= c.htt1)
                hai2 = c.hai11 + (c.hai00 - c.hai11) * g4_pow(
                    G4DIV(tairc - c.htt1, c.htt0 - c.htt1), c.haixp);
            if (tairc <= c.htt1) hai2 = c.hai11;
            const float dhai1 = hai2;
            if (qsg1 >= hai2)
                ftnsg = 1.0f - 0.80f * g4_min(g4_max(G4DIV(qsg1 - hai2, dhai1), 0.f), 1.f);
        }
    }
    return ftnsg;
}

// ---------------------------------------------------------------------------
// vqrqi (:1996-2063): rain (isg=1) or cloud-ice (isg=2) fall speed, cm/s
// ---------------------------------------------------------------------------
__device__ float g4_vqrqi(const G4K &c, int isg, float r00, float fv,
                          float qri, float ql, float tair)
{
    float y1 = r00 * qri;
    float ww1 = 0.f;
    if (y1 > c.cmin) {
        if (isg == 1) {
            float ftnw = 1.f;
            if (ql < c.cmin && tair > c.t0) {
                float bin_factor = 0.11f * g4_pow(1000.f * qri, -1.27f) + 0.98f;
                bin_factor = g4_min(bin_factor, 1.30f);
                ftnw = G4DIV(1.f, g4_pow(bin_factor, 3.35f));
                const float ftnwmin = G4DIV(r00 * qri, c.draimax);
                if (qri <= 0.001f) ftnw = g4_max(ftnw, G4DIV(ftnwmin, c.tnw));
            }
            const float vs = G4SQRT(y1);
            const float vg = G4SQRT(vs);
            const float vr = c.vrc0
                + G4DIV(c.vrc1 * vg, g4_pow(ftnw, 0.25f))
                + G4DIV(c.vrc2 * vs, g4_pow(ftnw, 0.50f))
                + G4DIV(c.vrc3 * vg * vs, g4_pow(ftnw, 0.75f));
            ww1 = g4_max(fv * vr, 0.e0f);
        } else if (isg == 2) {
            y1 = 1.e6f * r00 * qri;
            if (y1 > 1.e-6f) {
                y1 = y1 * 1.e-3f;
                const float const_m = G4DIV(1.f, 5.38e7f);
                const float bb1 = const_m * g4_pow(y1, 0.25f);
                const float bb2 = 11.9f * g4_pow(bb1, 0.5f);
                ww1 = g4_max(1.49e4f * g4_pow(bb2, 1.31f), 0.0f);
                ww1 = ww1 * 100.f;
                if (ww1 > 50.f) ww1 = 50.f;
            }
        }
    }
    return ww1;
}

// eff_rad (:4956-4991)
__device__ __forceinline__ float g4_eff_rad(float lambda)
{
    if (lambda <= 0.e0f || isnan(lambda)) return 0.e0f;
    return G4DIV(1.5e0f, lambda * 100.f) * 1.0e+6f;
}

#define G4AT(a, k) (a)[(size_t)(k) * ncol + col]

// ---------------------------------------------------------------------------
// fall_flux (:570-1265) for one column.  The flux loop of each species is a
// forward substep sequence; vt holds the current substep's speeds.
// ---------------------------------------------------------------------------
__device__ void g4_sediment(const G4K &c, int species, float *q,
                            const float *qc_cloud, const float *qg_orig,
                            const float *qh_orig, const float *qs_cur,
                            const float *th, const float *pii,
                            const float *rho, const float *z,
                            const float *dz8w, float topo, float *vt,
                            float *sed, float dtb, float *ppt,
                            int nz, int ncol, int col)
{
    // species: 0 rain, 1 snow, 2 graupel, 3 hail, 4 cloud ice
    float t_del_tv = 0.f;
    float del_tv = dtb;
    bool notlast = true;
    const int kte = nz - 1;
    while (notlast) {
        int min_q = kte;
        int max_q = -1;
        for (int k = 0; k <= kte - 1; ++k) {
            G4AT(vt, k) = 0.f;
            const float qk = G4AT(q, k);
            const float rhoz = G4AT(rho, k);
            const float r00 = rhoz * 0.001f;
            const float fv = G4SQRT(G4DIV(1.29f, rhoz));
            if (qk > c.cmin) {
                min_q = min(min_q, k);
                max_q = max(max_q, k);
                float v = 0.f;
                if (species == 0) {
                    const float tair = G4AT(th, k) * G4AT(pii, k);
                    v = g4_vqrqi(c, 1, r00, fv, qk, G4AT(qc_cloud, k), tair);
                    v = v * 0.01f;
                    G4AT(vt, k) = v;
                    if (!(v > 0.0f)) continue;   // EMK NUWRF bug fix (:815)
                } else if (species == 1) {
                    const float y1 = qk;
                    const float vscf = c.vsc * fv;
                    const float tair = G4AT(th, k) * G4AT(pii, k);
                    const float tairc = tair - c.t0;
                    float qhz2 = G4AT(qh_orig, k), qgz2 = G4AT(qg_orig, k);
                    if (k + 1 < kte + 1 - 2 && tairc >= -5.f) {
                        qhz2 = G4AT(qh_orig, k + 1);
                        qgz2 = G4AT(qg_orig, k + 1);
                    }
                    const float ftns0 = g4_sgmap(c, 1, qk, G4AT(qg_orig, k), qgz2,
                                                 G4AT(qh_orig, k), qhz2, r00, tairc);
                    const float ftns = g4_pow(ftns0, c.bsq);
                    const float fros0 = g4_sgmap(c, 3, qk, G4AT(qg_orig, k), qgz2,
                                                 G4AT(qh_orig, k), qhz2, r00, tairc);
                    const float fros = g4_pow(fros0, c.bsq);
                    v = g4_max(G4DIV(G4DIV(vscf * g4_pow(r00 * y1, c.bsq), ftns), fros), 0.e0f);
                    v = v * 0.01f;
                    G4AT(vt, k) = v;
                } else if (species == 2) {
                    const float y1 = qk;
                    const float vgcr = c.vgc * fv;
                    const float vgcr2 = c.vgc2 * fv;
                    const float tair = G4AT(th, k) * G4AT(pii, k);
                    const float tairc = tair - c.t0;
                    float qhz2 = G4AT(qh_orig, k), qgz2 = qk;
                    if (k + 1 < kte + 1 - 2 && tairc >= -5.f) {
                        qhz2 = G4AT(qh_orig, k + 1);
                        qgz2 = G4AT(q, k + 1);
                    }
                    const float ftng0 = g4_sgmap(c, 2, G4AT(qs_cur, k), qk, qgz2,
                                                 G4AT(qh_orig, k), qhz2, r00, tairc);
                    float ftng = g4_pow(ftng0, c.bgq);
                    v = g4_max(G4DIV(vgcr * g4_pow(r00 * y1, c.bgq), ftng), 0.0f);
                    v = v * 0.01f;
                    if (y1 > c.qrog2) {
                        ftng = g4_pow(ftng0, c.bgq2);
                        v = g4_max(G4DIV(vgcr2 * g4_pow(r00 * y1, c.bgq2), ftng), 0.0f);
                        v = v * 0.01f;
                    }
                    G4AT(vt, k) = v;
                } else if (species == 3) {
                    const float y1 = qk;
                    const float vhcr = G4DIV(c.vhc, G4SQRT(r00));
                    v = g4_max(vhcr * g4_pow(y1 * r00, c.bhq), 0.e0f);
                    v = v * 0.01f;
                    G4AT(vt, k) = v;
                } else {
                    v = 0.f;
                    float y1 = rhoz * 1000.f * qk;
                    if (y1 >= 1.e-6f) {
                        const float tair = G4AT(th, k) * G4AT(pii, k);
                        v = g4_vqrqi(c, 2, r00, fv, qk, G4AT(qc_cloud, k), tair);
                    }
                    v = v * 0.01f;
                    G4AT(vt, k) = v;
                    if (!(v > 1.e-20f)) continue;   // EMK (:1176)
                }
                const float dz = (k == 0) ? G4AT(z, 0) - topo
                                          : G4AT(z, k) - G4AT(z, k - 1);
                del_tv = g4_min(del_tv, G4DIV(0.9f * dz, v));
            }
        }
        if (max_q >= min_q) {
            t_del_tv = t_del_tv + del_tv;
            if (t_del_tv >= dtb) {
                notlast = false;
                del_tv = dtb + del_tv - t_del_tv;
            }
            float fluxin = 0.f;
            for (int k = max_q; k >= min_q; --k) {
                const float rhoz = G4AT(rho, k);
                float qz = G4AT(q, k);
                const float fluxout = rhoz * G4AT(vt, k) * qz;
                const float flux = G4DIV(G4DIV(fluxin - fluxout, rhoz), G4AT(dz8w, k));
                qz = qz + del_tv * flux;
                qz = g4_max(0.f, qz);
                G4AT(q, k) = qz;
                fluxin = fluxout;
                G4AT(sed, k) = G4AT(sed, k) + fluxin;
            }
            if (min_q == 0) {
                *ppt = *ppt + fluxin * del_tv;
            } else {
                float qz = G4AT(q, min_q - 1);
                qz = qz + G4DIV(G4DIV(del_tv * fluxin, G4AT(rho, min_q - 1)),
                                G4AT(dz8w, min_q - 1));
                qz = g4_max(0.f, qz);
                G4AT(q, min_q - 1) = qz;
            }
        } else {
            notlast = false;
        }
    }
}

// ---------------------------------------------------------------------------
// The column: fall_flux, then saticel_s level by level (:2548-4470).
// saticel_s reads qgwrf/qhwrf at k+1 (its own INPUT, i.e. the post-fall_flux
// value) and writes every level back only after the level loop; writing
// level k in place at its end is the same, because nothing after level k
// reads level k's input again and level k+1 is not yet written.
// ---------------------------------------------------------------------------
extern "C" __global__ void gsfc4ice_column(
    const float *__restrict__ table,
    float *th, float *qv, float *ql, float *qr, float *qi, float *qs,
    float *qg, float *qh,
    const float *__restrict__ rho, const float *__restrict__ pii,
    const float *__restrict__ p, const float *__restrict__ z,
    const float *__restrict__ dz8w, const float *__restrict__ w,
    const float *__restrict__ ht, const float *__restrict__ xland,
    float *refl_10cm,
    float *physc, float *physe, float *physd, float *physs, float *physm,
    float *physf,
    float *acphysc, float *acphyse, float *acphysd, float *acphyss,
    float *acphysm, float *acphysf,
    float *re_cloud, float *re_rain, float *re_ice, float *re_snow,
    float *re_graupel, float *re_hail,
    float *preci3d, float *precs3d, float *precg3d, float *prech3d,
    float *precr3d,
    float *rainnc, float *rainncv, float *snownc, float *snowncv, float *sr,
    float *graupelnc, float *graupelncv, float *hailnc, float *hailncv,
    float *vt, float *err,
    float dt, float dx, int itimestep, int refl_due, int nz, int ncol)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const G4K &c = *reinterpret_cast<const G4K *>(table);

    // ------------------------------------------------------------ fall_flux
    for (int k = 0; k < nz; ++k) {
        G4AT(preci3d, k) = 0.f; G4AT(precs3d, k) = 0.f; G4AT(precg3d, k) = 0.f;
        G4AT(prech3d, k) = 0.f; G4AT(precr3d, k) = 0.f;
    }
    float pptrain = 0.f, pptsnow = 0.f, pptgraul = 0.f, ppthail = 0.f,
          pptice = 0.f;
    const float topo = ht[col];
    g4_sediment(c, 0, qr, ql, qg, qh, qs, th, pii, rho, z, dz8w, topo, vt,
                precr3d, dt, &pptrain, nz, ncol, col);
    g4_sediment(c, 1, qs, ql, qg, qh, qs, th, pii, rho, z, dz8w, topo, vt,
                precs3d, dt, &pptsnow, nz, ncol, col);
    g4_sediment(c, 2, qg, ql, qg, qh, qs, th, pii, rho, z, dz8w, topo, vt,
                precg3d, dt, &pptgraul, nz, ncol, col);
    g4_sediment(c, 3, qh, ql, qg, qh, qs, th, pii, rho, z, dz8w, topo, vt,
                prech3d, dt, &ppthail, nz, ncol, col);
    g4_sediment(c, 4, qi, ql, qg, qh, qs, th, pii, rho, z, dz8w, topo, vt,
                preci3d, dt, &pptice, nz, ncol, col);
    snowncv[col] = pptsnow;
    snownc[col] = snownc[col] + pptsnow;
    graupelncv[col] = pptgraul;
    graupelnc[col] = graupelnc[col] + pptgraul;
    hailncv[col] = ppthail;
    hailnc[col] = hailnc[col] + ppthail;
    const float rncv = pptrain + pptsnow + pptgraul + pptice + ppthail;
    rainncv[col] = rncv;
    rainnc[col] = rainnc[col] + pptrain + pptsnow + pptgraul + pptice + ppthail;
    float srv = 0.f;
    if (rncv > 0.f) srv = G4DIV(pptsnow + pptgraul + pptice + ppthail, rncv);
    sr[col] = srv;

    // ------------------------------------------------------------ saticel_s
    const float dwv0 = 0.226f;
    const float d2t = dt;
    const float r2ig = 1.f, r2is = 1.f, r2ih = 1.f;
    const float cmin = 1.e-20f, cmin1 = 1.e-20f;
    const float xssi = 0.16f, rssi = 0.05f, xsubi = 0.70f;
    const float xnsplnt = 370.f, xmsplnt = 4.4e-8f;
    const float hmtemp1 = -2.f, hmtemp2 = -4.f, hmtemp3 = -6.f, hmtemp4 = -8.f;
    const float bs3 = c.bs + 3.f, bg3 = c.bg + 3.f, bh3 = c.bh + 3.f;
    const float bsh5 = 2.5f + c.bsh, bgh5 = 2.5f + c.bgh, bhh5 = 2.5f + c.bhh;
    const float bs6 = c.bs + 6.f;
    const float r10t = c.rn10 * d2t, r25a = c.rn25;
    const float r11t = c.rn11 * d2t, r19t = c.rn19 * d2t, r19at = c.rn19a * d2t;
    const float r20t = c.rn20 * d2t, r23t = c.rn23 * d2t, r30t = c.rn30 * d2t;
    const float r33t = c.rn33 * d2t;
    const float bg3_2 = c.bg2 + 3.f, bgh5_2 = 2.5f + c.bgh2;
    const float thresh_evap = -(39.974f * g4_exp(-G4DIV(1.194f * dx, 1000.f)));
    const float Rc = 1.e-3f, Ra = 1.e-5f, Cna = 500.f;
    // 5.0e-6*exp(0.304*40.0) is literals only: gfortran folds the whole
    // product at compile time with a correctly rounded EXP (0x483a849f), so
    // the referee carries this word, not a runtime expf.
    const float c_nci_cap = g4_bits(0x3f747912u);
    const float qb0 = 0.f, tb0 = 0.f;
    const float tdiff = c.t00 - c.t0;
    const bool land = xland[col] == 1.0f;
    const bool water = xland[col] == 2.0f;
    if (!land && !water) {
        // WRF stops the run here (wrf_error_fatal, :4417-4419); the host
        // refuses such a mask before launch, this is the device's own net.
        err[0] = 1.f;
    }
    const float rho1 = rho[col] * 0.001f;
    const bool zero_acc = (itimestep == 1);

    for (int k = 0; k < nz; ++k) {
        const float rhok = G4AT(rho, k) * 0.001f;
        const float p0 = G4AT(p, k) * 10.0f;
        const float pi = G4AT(pii, k);
        const float ww1 = G4AT(w, k) * 100.f;
        const float fvk = G4SQRT(G4DIV(rho1, rhok));
        const float qi_in = G4AT(qi, k);

        const float rp0 = G4DIV(3.799052e3f, p0);
        const float pi0 = pi;
        const float pir = G4DIV(1.f, pi);
        const float pr0 = G4DIV(1.f, p0);
        const float r00 = rhok;
        const float rr0 = G4DIV(1.f, rhok);
        const float rrs = G4SQRT(rr0);
        const float rrq = G4SQRT(rrs);
        const float fv0 = fvk;
        const float fvs = G4SQRT(fvk);
        const float zrr = 1.e5f * c.zrc * rrq;
        const float zsr = 1.e5f * c.zsc * rrq;
        const float zgr = 1.e5f * c.zgc * rrq;
        const float zhr = 1.e5f * c.zhc * rrq;
        const float cp409 = c.c409 * pi0;
        const float cv409 = c.c409 * c.avc;
        const float cp580 = c.c580 * pi0;
        const float cs580 = c.c580 * c.asc;
        const float alvr = r00 * c.alv;
        const float afcp = c.afc * pir;
        const float avcp = c.avc * pir;
        const float ascp = c.asc * pir;
        const float vscf = c.vsc * fv0;
        const float vgcf = c.vgc * fv0;
        const float vgcf2 = c.vgc2 * fv0;
        const float vhcr = c.vhc * rrs;
        const float dwvp = c.c879 * pr0;
        const float r3f = c.rn3 * fv0, r4f = c.rn4 * fv0, r5f = c.rn5 * fv0,
                    r6f = c.rn6 * fv0;
        const float r12r = c.rn12 * r00;
        const float r14f = c.rn14 * fv0, r14f2 = c.rn142 * fv0;
        const float r15f = c.rn15 * fv0, r15af = c.rn15a * fv0,
                    r15af2 = c.rn15a2 * fv0;
        const float r18r = c.rn18 * rr0;
        const float r22f = c.rn22 * fv0;
        const float r25rt = c.rn25 * rr0 * d2t;
        const float r32rt = c.rn32 * d2t * rrs;
        const float r7rf = c.rn7 * rr0 * fv0, r8rf = c.rn8 * rr0 * fv0,
                    r9rf = c.rn9 * rr0 * fv0, r16rf = c.rn16 * rr0 * fv0;
        const float r101r = c.rn101 * rr0;
        const float r102rf = c.rn102 * rrs * fvs;
        const float r191r = c.rn191 * rr0;
        const float r192rf = c.rn192 * rrs * fvs;
        const float r192rf2 = c.rn192_2 * rrs * fvs;
        const float r331r = c.rn331 * rr0;
        const float r332rf = c.rn332 * rrs * fvs;
        const float r34f = c.rn34 * fv0;
        const float r231r = c.rn231 * rr0;
        const float r232rf = c.rn232 * rrs * fvs;
        const float h9r = c.hn9 * rr0, h10r = c.hn10 * rr0;
        const float h14r = c.hn14 * rrs, h15ar = c.hn15a * rrs;
        const float h16r = c.hn16 * rr0, h17r = c.hn17 * rr0;
        const float h17aq = c.hn17a * rrq, h19aq = c.hn19a * rrq;
        const float h19rt = c.hn19 * rr0 * d2t;
        const float h10ar = c.hn10a * r00;
        const float h20t = c.hn20 * d2t;
        const float h20bq = c.hn20b * rrq;

        float pt = G4AT(th, k);
        float qvv = G4AT(qv, k);
        float qcv = G4AT(ql, k);
        float qrv = G4AT(qr, k);
        float qiv = qi_in;
        float qsv = G4AT(qs, k);
        float qgv = G4AT(qg, k);
        float qhv = G4AT(qh, k);
        if (qcv <= cmin) qcv = 0.0f;
        if (qrv <= cmin) qrv = 0.0f;
        if (qiv <= cmin) qiv = 0.0f;
        if (qsv <= cmin) qsv = 0.0f;
        if (qgv <= cmin) qgv = 0.0f;
        if (qhv <= cmin) qhv = 0.0f;
        float tair = (pt + tb0) * pi0;
        float tairc = tair - c.t0;
        float zr = zrr, zs = zsr, zg = zgr, zh = zhr;
        float vr = 0.f, vs = 0.f, vg = 0.f, vi = 0.f, vh = 0.f;
        float ftns = 1.f, ftng = 1.f, ftns0 = 1.f, ftng0 = 1.f, fros = 1.f,
              ftnh = 1.f, ftnh0 = 1.f, fros0 = 1.f;
        float cnd = 0.f, dep = 0.f, ern = 0.f, pint = 0.f, pidep = 0.f;
        float psdep = 0.f, pgdep = 0.f, dd1 = 0.f, dd = 0.f, pgsub = 0.f;
        float psmlt = 0.f, pgmlt = 0.f, pimlt = 0.f, psacw = 0.f, piacr = 0.f;
        float pssub = 0.f, psfw = 0.f, psfi = 0.f, pgfr = 0.f, psacr = 0.f;
        float wgacr = 0.f, pihom = 0.f, pidw = 0.f, psaut = 0.f, psaci = 0.f;
        float praci = 0.f, pwacs = 0.f, qsacw = 0.f, pracs = 0.f, qracs = 0.f;
        float qsacr = 0.f, praut = 0.f, pracw = 0.f, pgacs = 0.f;
        float qgacw = 0.f, dgaci = 0.f, dgacs = 0.f, wgacs = 0.f, wgaci = 0.f;
        float dgacw = 0.f, dgacr = 0.f, pgwet = 0.f, qgacr = 0.f, pimm = 0.f;
        float pcfr = 0.f, pihms = 0.f, pihmg = 0.f, pmlts = 0.f, pmltg = 0.f;
        float phfr = 0.f, phmlt = 0.f, dhacw = 0.f, qhacw = 0.f, dhacr = 0.f;
        float qhacr = 0.f, whacr = 0.f, whaci = 0.f, whacs = 0.f, whacg = 0.f;
        float phwet = 0.f, phdep = 0.f, phsub = 0.f, pvapg = 0.f, pvaph = 0.f;
        float primh = 0.f, pihmh = 0.f, pracg = 0.f, qracg = 0.f, pg2h = 0.f;
        float dlt1 = 0.f, dlt2 = 0.f, dlt3 = 0.f, dlt4 = 0.f;
        float pr = 0.f, ps = 0.f, pg = 0.f, prn = 0.f, psn = 0.f;
        float qsw = 0.f, qsi = 0.f, ssi = 0.f, esw = 0.f, ssw = 0.f;
        float esi = 0.f, rtair = 0.f, dm = 0.f, rsub1 = 0.f, dda0 = 0.f;
        float y1, y2, y3, y4, y5;
        float fssi, wssi, r_nci, c_nci, term1, term2, fdwv, ftnw, ftnwmin;
        float bin_factor, del, temp;
        float qhz2, qgz2;
        int it;

        y1 = G4DIV(c.c149 * g4_pow(tair, 1.5f), tair + 120.f);
        const float dwv = dwvp * g4_pow(tair, 1.81f);
        const float tca = c.c141 * y1;
        const float scv = G4DIV(1.f, g4_pow(rr0 * y1, .1666667f) * g4_pow(dwv, .3333333f));

        // ZR, ZS, ZG, ZH, VR, VS, VG, VI, VH (:2798-2869)
        if (qrv > cmin) {
            dd = r00 * qrv;
            y1 = G4SQRT(dd);
            y2 = G4SQRT(y1);
            zr = G4DIV(c.zrc, y2);
        }
        vr = g4_vqrqi(c, 1, r00, fv0, qrv, qcv, tair);
        vi = g4_vqrqi(c, 2, r00, fv0, qiv, qcv, tair);
        ftns = 1.f;
        ftns0 = 1.f;
#define G4_SELECT_Z2()                                                        \
        qhz2 = G4AT(qh, k);                                                   \
        qgz2 = G4AT(qg, k);                                                   \
        if (k + 1 < nz - 2 && tairc >= -5.f) {                                \
            qhz2 = G4AT(qh, k + 1);                                           \
            qgz2 = G4AT(qg, k + 1);                                           \
        }
        G4_SELECT_Z2();
        ftns0 = g4_sgmap(c, 1, qsv, qgv, qgz2, qhv, qhz2, r00, tairc);
        fros0 = g4_sgmap(c, 3, qsv, qgv, qgz2, qhv, qhz2, r00, tairc);
        if (qsv > cmin) {
            dd = r00 * qsv;
            y1 = g4_pow(dd, .25f);
            ftns = g4_pow(ftns0, 0.25f);
            fros = g4_pow(fros0, 0.25f);
            zs = G4DIV(c.zsc, y1) * ftns * fros;
            ftns = g4_pow(ftns0, c.bsq);
            fros = g4_pow(fros0, c.bsq);
            vs = g4_max(G4DIV(G4DIV(vscf * g4_pow(dd, c.bsq), ftns), fros), 0.f);
        }
        ftng = 1.f;
        ftng0 = 1.f;
        ftng0 = g4_sgmap(c, 2, qsv, qgv, qgz2, qhv, qhz2, r00, tairc);
        if (qgv > cmin) {
            dd = r00 * qgv;
            y1 = g4_pow(dd, .25f);
            ftng = g4_pow(ftng0, 0.25f);
            zg = G4DIV(c.zgc, y1) * ftng;
            if (dd > c.qrog2) zg = G4DIV(c.zgc2, y1) * ftng;
            ftng = g4_pow(ftng0, c.bgq);
            vg = g4_max(G4DIV(vgcf * g4_pow(dd, c.bgq), ftng), 0.0f);
            if (dd > c.qrog2) {
                ftng = g4_pow(ftng0, c.bgq2);
                vg = g4_max(G4DIV(vgcf2 * g4_pow(dd, c.bgq2), ftng), 0.e0f);
            }
        }
        ftnh0 = g4_sgmap(c, 4, qsv, qgv, qgz2, qhv, qhz2, r00, tairc);
        if (qhv > cmin) {
            dd = r00 * qhv;
            y1 = g4_pow(dd, .25f);
            ftnh = g4_pow(ftnh0, 0.25f);
            zh = G4DIV(c.zhc, y1) * ftnh;
            ftnh = g4_pow(ftnh0, c.bhq);
            vh = g4_max(G4DIV(vhcr * g4_pow(dd, c.bhq), ftnh), 0.0f);
        }
        if (qrv <= cmin1) vr = 0.0f;
        if (qsv <= cmin1) vs = 0.0f;
        if (qgv <= cmin1) vg = 0.0f;
        if (qiv <= cmin1) vi = 0.0f;
        if (qhv <= cmin1) vh = 0.0f;

        // PSAUT, PSACI, PSACW, PRACI, PIACR, PWACS (:2885-2951)
        pihms = 0.f; pihmg = 0.f; psaut = 0.f; psaci = 0.f; praci = 0.f;
        piacr = 0.f; psacw = 0.f; pwacs = 0.f; qsacw = 0.f;
        ftns = 1.f; ftng = 1.f; ftnh = 1.f;
        ftns = ftns0;
        ftng = ftng0;
        if (tair < c.t0) {
            const float rn1 = G4DIV(1.f, 300.f);
            const float bnd1 = 6.e-5f;
            esi = 0.25f;
            psaut = r2is * g4_max(rn1 * esi * (qiv - bnd1 * fv0 * fv0), 0.0f);
            ftns = ftns0;
            ftng = ftng0;
            fros = fros0;
            ftnh = ftnh0;
            esi = 1.0f;
            const float dmicrons = g4_pow(G4DIV(G4DIV(G4DIV(r00 * qsv, c.roqs * fros),
                                                      c.cpi), c.tns * ftns), .25f) * 1.e4f;
            esi = g4_min(1.f, g4_pow(G4DIV(dmicrons, 375.f), 3.f));
            y1 = 1.0f;
            if (vs > 0.f) y1 = fabsf(G4DIV(vs - vi, vs));
            psaci = G4DIV(r2is * y1 * r3f * qiv, g4_pow(zs, bs3)) * ftns * esi;
            if (qsv <= cmin) psaci = 0.f;
            psacw = G4DIV(r2is * r4f * qcv, g4_pow(zs, bs3)) * ftns;
            if (qsv <= cmin) psacw = 0.f;
            y2 = 0.f;
            if ((tairc <= hmtemp1) && (tairc >= hmtemp4)) y2 = 0.5f;
            if ((tairc <= hmtemp2) && (tairc >= hmtemp3)) y2 = 1.f;
            pihms = r2ih * psacw * y2 * xnsplnt * 1000.f * xmsplnt;
            psacw = psacw - pihms;
            pwacs = G4DIV(r2is * r34f * qcv, g4_pow(zs, bs6)) * ftns * fros;
            if (qsv <= cmin) pwacs = 0.f;
            y1 = G4DIV(1.f, zr);
            y2 = y1 * y1;
            y3 = y1 * y2;
            y5 = 1.0f;
            if (vr > 0.f) y5 = fabsf(G4DIV(vr - vi, vr));
            dd = y5 * r5f * qiv * y3 * (c.rn50 + c.rn51 * y1 + c.rn52 * y2 + c.rn53 * y3);
            praci = g4_max(dd, 0.0f);
            if (qrv <= cmin) praci = 0.f;
            y4 = y3 * y3;
            dd1 = y5 * r6f * qiv * y4 * (c.rn60 + c.rn61 * y1 + c.rn62 * y2 + c.rn63 * y3);
            piacr = g4_max(dd1, 0.0f);
            if (qrv <= cmin) piacr = 0.f;
        } else {
            qsacw = G4DIV(r2is * r4f * qcv, g4_pow(zs, bs3)) * ftns;
            if (qsv <= cmin) psacw = 0.f;
        }

        // PRAUT, PRACW (:2988-2998)
        praut = g4_max(c.rn21 * (qcv - c.bnd21), 0.0f);
        y1 = G4DIV(1.f, zr);
        y2 = y1 * y1;
        y3 = y1 * y2;
        y4 = r22f * qcv * y3 * (c.rn50 + c.rn51 * y1 + c.rn52 * y2 + c.rn53 * y3);
        pracw = g4_max(y4, 0.0f);
        if (qrv <= cmin) pracw = 0.f;
        pidep = 0.0f;
        psfw = 0.0f;
        psfi = 0.0f;

        // PSFW, PSFI (:3011-3107)
        if (tair < c.t0) {
            y1 = g4_max(g4_min(tairc, -1.f), -31.f);
            it = (int)fabsf(y1);
            y1 = c.rn12a[it - 1];
            y2 = c.rn12b[it - 1];
            y3 = c.rn13[it - 1];
            psfw = r2is * g4_max(d2t * y1 * (y2 + r12r * qcv) * qiv, 0.0f);
            psfi = r2is * y3 * qiv;
            y4 = G4DIV(1.f, tair - c.c358);
            y5 = G4DIV(1.f, tair - c.c76);
            qsw = rp0 * g4_exp(c.c172 - c.c409 * y4);
            qsi = rp0 * g4_exp(c.c218 - c.c580 * y5);
            float hfact = G4DIV(qvv + qb0 - qsi, qsw - qsi + cmin1);
            if (hfact > 1.f) hfact = 1.f;
            float sfact = 1.f;
            ssi = G4DIV(qvv + qb0, qsi) - 1.f;
            fssi = rssi;
            wssi = G4DIV(ww1, 100.f) - 2.0f;
            if (wssi > 0.f) fssi = rssi + g4_min(xssi, wssi * 0.01f);
            fssi = g4_min(ssi, fssi);
            if (tairc < -40.0f) c_nci = c_nci_cap;
            else c_nci = 5.0e-6f * g4_exp(0.304f * fabsf(tairc));
            r_nci = c_nci;
            r_nci = g4_max(r_nci, G4DIV(r00 * qiv, c.ami50));
            dd = g4_min(G4DIV(r00 * qiv, r_nci), c.ami40);
            const float yy1 = 1.f - c.aa2[it - 1];
            sfact = G4DIV(g4_pow(c.ami50, yy1) - g4_pow(c.ami40, yy1),
                          g4_pow(c.ami50, yy1) - g4_pow(dd, yy1));
            esi = G4DIV(qsi, rp0) * c.c610;
            y3 = G4DIV(1.f, tair);
            term1 = y3 * (c.rn20a * y3 - c.rn20b);
            term2 = G4DIV(c.rn10c * tair, esi);
            dd = term1 + term2;
            fdwv = G4DIV(dd, term1 + G4DIV(term2 * dwv0, dwv));
            psfw = 0.0f;
            if (hfact > 0.f) psfi = r2is * psfi * hfact * sfact * fdwv;
            else psfi = 0.f;
            if (qiv <= 1.e-5f * fv0 * fv0) psfi = 0.0f;
        }

        // PGACS, WGACS, WHACS, WHACG, DGACW, DHACW, DGACI, WGACI, WHACI,
        // DGACR, PRACG, DHACR, PGWET, PHWET (:3123-3317)
        y1 = fabsf(vg - vs);
        y2 = zs * zg;
        y3 = G4DIV(5.f, y2);
        y4 = .08f * y3 * y3;
        y5 = .05f * y3 * y4;
        y2 = y1 * (G4DIV(y3, g4_p5(zs)) + G4DIV(y4, g4_p3(zs)) + G4DIV(y5, zs));
        pgacs = r2ig * r2is * r9rf * y2 * ftns * ftng * fros;
        if (qsv <= cmin) pgacs = 0.f;
        if (qgv <= cmin) pgacs = 0.f;
        dgacs = pgacs;
        dgacs = 0.0f;
        wgacs = 0.0f;
        wgacs = 10.f * r9rf * y2 * ftns * ftng * fros;
        if (qgv <= cmin) wgacs = 0.f;
        if (qsv <= cmin) wgacs = 0.f;

        whacs = 0.0f;
        y1 = fabsf(vh - vs);
        y2 = zs * zh;
        y3 = G4DIV(5.f, y2);
        y4 = .08f * y3 * y3;
        y5 = .05f * y3 * y4;
        dd = y1 * (G4DIV(y3, g4_p5(zs)) + G4DIV(y4, g4_p3(zs)) + G4DIV(y5, zs));
        whacs = r2ih * r2is * g4_min(h9r * dd * ftnh * ftns * fros, G4DIV(qsv, d2t));
        if (qsv <= cmin) whacs = 0.f;
        if (qhv <= cmin) whacs = 0.f;

        whacg = 0.0f;
        y1 = fabsf(vh - vg);
        y2 = zg * zh;
        y3 = G4DIV(5.f, y2);
        y4 = .08f * y3 * y3;
        y5 = .05f * y3 * y4;
        dd = y1 * (G4DIV(y3, g4_p5(zg)) + G4DIV(y4, g4_p3(zg)) + G4DIV(y5, zg));
        whacg = r2ih * r2ig * g4_min(h10r * dd * ftnh * ftng, G4DIV(qgv, d2t));
        if (r00 * qgv > c.qrog2) whacg = G4DIV(whacg, c.roqg) * c.roqg2 * 0.5f;
        if (qgv <= cmin) whacg = 0.f;
        if (qhv <= cmin) whacg = 0.f;

        y1 = G4DIV(1.f, g4_pow(zg, bg3));
        esi = 1.0f;
        float dmicrong = g4_pow(G4DIV(G4DIV(G4DIV(r00 * qgv, c.roqg), c.cpi),
                                      c.tng * ftng), .25f) * 1.e4f;
        if (r00 * qgv > c.qrog2) {
            y1 = G4DIV(1.f, g4_pow(zg, bg3_2));
            dmicrong = g4_pow(G4DIV(G4DIV(G4DIV(r00 * qgv, c.roqg2), c.cpi),
                                    c.tng * ftng), .25f) * 1.e4f;
        }
        esi = g4_min(1.f, g4_pow(G4DIV(dmicrong, 500.f), 1.1f));
        dgacw = r2ig * esi * r14f * qcv * y1 * ftng;
        if (r00 * qgv > c.qrog2) dgacw = r2ig * esi * r14f2 * qcv * y1 * ftng;
        if (qgv <= cmin) dgacw = 0.f;

        dhacw = 0.0f;
        y2 = G4DIV(1.f, g4_pow(zh, bh3));
        dhacw = r2ih * g4_max(h14r * qcv * y2 * ftnh, 0.0f);
        if (qhv <= cmin) dhacw = 0.f;
        y2 = 0.f;
        if ((tairc <= hmtemp1) && (tairc >= hmtemp4)) y2 = 0.5f;
        if ((tairc <= hmtemp2) && (tairc >= hmtemp3)) y2 = 1.f;
        pihmg = r2ig * dgacw * y2 * xnsplnt * 1000.f * xmsplnt;
        dgacw = r2ig * (dgacw - pihmg);
        pihmh = 0.0f;
        pihmh = r2ih * dhacw * y2 * xnsplnt * 1000.f * xmsplnt;
        dhacw = r2ih * (dhacw - pihmh);

        qgacw = r2ig * dgacw;
        qhacw = r2ih * dhacw;
        y1 = G4DIV(1.f, g4_pow(zg, bg3));
        y5 = 1.0f;
        if (vg > 0.f) y5 = fabsf(G4DIV(vg - vi, vg));
        dgaci = 0.0f;
        wgaci = 0.0f;
        wgaci = y5 * r15af * qiv * y1 * ftng;
        if (r00 * qgv > c.qrog2) {
            y1 = G4DIV(1.f, g4_pow(zg, bg3_2));
            wgaci = y5 * r15af2 * qiv * y1 * ftng;
        }
        if (qgv <= cmin) wgaci = 0.f;

        whaci = 0.0f;
        y5 = 1.0f;
        if (vh > 0.f) y5 = fabsf(G4DIV(vh - vi, vh));
        y2 = G4DIV(1.f, g4_pow(zh, bh3));
        whaci = r2ih * g4_min(y5 * h15ar * qiv * y2 * ftnh, G4DIV(qiv, d2t));
        if (qhv <= cmin) whaci = 0.f;

        y1 = fabsf(vg - vr);
        y2 = zr * zg;
        y3 = G4DIV(5.f, y2);
        y4 = .08f * y3 * y3;
        y5 = .05f * y3 * y4;
        dd = r16rf * y1 * (G4DIV(y3, g4_p5(zr)) + G4DIV(y4, g4_p3(zr)) + G4DIV(y5, zr)) * ftng;
        dgacr = r2ig * g4_max(dd, 0.0f);
        if (qgv <= cmin) dgacr = 0.f;
        if (qrv <= cmin) dgacr = 0.f;
        qgacr = dgacr;

        pracg = 0.f;
        qracg = 0.f;
        y2 = zr * zg;
        y3 = G4DIV(5.f, y2);
        y4 = .08f * y3 * y3;
        y5 = .05f * y3 * y4;
        pracg = G4DIV(r16rf, c.roqr) * c.roqg * y1
              * (G4DIV(y3, g4_p5(zg)) + G4DIV(y4, g4_p3(zg)) + G4DIV(y5, zg)) * ftng;
        if (r00 * qgv > c.qrog2) pracg = G4DIV(pracg, c.roqg) * c.roqg2;
        if (qgv <= cmin) pracg = 0.f;
        if (qrv <= cmin) pracg = 0.f;
        qracg = g4_min(d2t * pracg, qgv);

        y1 = fabsf(vh - vr);
        y2 = zr * zh;
        y3 = G4DIV(5.f, y2);
        y4 = .08f * y3 * y3;
        y5 = .05f * y3 * y4;
        dd = h16r * y1 * ftnh * (G4DIV(y3, g4_p5(zr)) + G4DIV(y4, g4_p3(zr)) + G4DIV(y5, zr));
        dhacr = r2ih * g4_max(dd, 0.0f);
        if (qhv <= cmin) dhacr = 0.f;
        if (qrv <= cmin) dhacr = 0.f;
        qhacr = dhacr;

        if (tair >= c.t0) {
            dgacs = 0.0f; wgacs = 0.0f; whacs = 0.0f; whacg = 0.0f;
            dgacw = 0.0f; dhacw = 0.0f; dgaci = 0.0f; wgaci = 0.0f;
            whaci = 0.0f; dgacr = 0.0f; pracg = 0.0f; dhacr = 0.0f;
        } else {
            pgacs = 0.0f; qgacw = 0.0f; qhacw = 0.0f; qgacr = 0.0f;
            qracg = 0.0f; qhacr = 0.0f;
        }

        pgwet = 0.0f;
        if (tair < c.t0) {
            y1 = G4DIV(1.f, c.alf + c.rn17c * g4_max(tairc, -75.0f));
            y2 = G4DIV(r191r, g4_p2(zg)) + G4DIV(r192rf, g4_pow(zg, bgh5));
            if (r00 * qgv > c.qrog2)
                y2 = (G4DIV(r191r, g4_p2(zg)) + G4DIV(r192rf2, g4_pow(zg, bgh5_2)));
            y4 = alvr * dwv * (rp0 - (qvv + qb0)) - tca * tairc;
            dd = y1 * (c.rn20 * ftng * y4 * y2 + (wgaci + wgacs) * (c.alf + c.rn17b * tairc));
            pgwet = g4_max(dd, 0.0f);
            if (qgv <= cmin) pgwet = 0.f;
        }

        phwet = 0.0f;
        if (tair < c.t0) {
            y1 = G4DIV(1.f, c.alf + c.rn17c * tairc);
            y3 = G4DIV(.78f, g4_p2(zh)) + G4DIV(h17aq * scv, g4_pow(zh, bhh5));
            y4 = alvr * dwv * (rp0 - (qvv + qb0)) - tca * tairc;
            dd = y1 * (h17r * ftnh * y4 * y3 + (whaci + whacs + whacg) * (c.alf + c.rn17b * tairc));
            phwet = r2ih * g4_max(dd, 0.0f);
            if (qhv <= cmin) phwet = 0.f;
        }

        // Negative cloud water (:3324-3360)
        y1 = G4DIV(qcv, d2t);
        psacw = g4_min(y1, psacw);
        pihms = g4_min(y1, pihms);
        praut = g4_min(y1, praut);
        pracw = g4_min(y1, pracw);
        psfw = g4_min(y1, psfw);
        dgacw = g4_min(y1, dgacw);
        pihmg = g4_min(y1, pihmg);
        dhacw = g4_min(y1, dhacw);
        pihmh = g4_min(y1, pihmh);
        qsacw = g4_min(y1, qsacw);
        qgacw = g4_min(y1, qgacw);
        qhacw = g4_min(y1, qhacw);
        y1 = d2t * (psacw + praut + pracw + psfw + dgacw + qsacw + qgacw
                    + pihms + pihmg + dhacw + qhacw + pihmh);
        qcv = qcv - y1;
        if (qcv < 0.0f) {
            y2 = 1.f;
            if (y1 != 0.f) y2 = G4DIV(qcv, y1) + 1.f;
            psacw = psacw * y2; praut = praut * y2; pracw = pracw * y2;
            psfw = psfw * y2; dgacw = dgacw * y2; dhacw = dhacw * y2;
            qsacw = qsacw * y2; qgacw = qgacw * y2; qhacw = qhacw * y2;
            pihms = pihms * y2; pihmg = pihmg * y2; pihmh = pihmh * y2;
            qcv = 0.0f;
        }

        // Wet graupel to hail; hail riming (:3365-3396)
        y1 = dgacw + dgacr;
        if (y1 < .95f * pgwet || y1 == 0.f) {
            wgaci = 0.0f;
            wgacs = 0.0f;
        }
        pg2h = 0.0f;
        if (y1 > 0.f && pgwet > 0.f && y1 > 1.0f * pgwet) {
            pg2h = G4DIV(qgv, d2t);
            pg2h = g4_min(pg2h, G4DIV(qgv, d2t));
        }
        whacr = phwet - dhacw - whaci - whacs - whacg;
        y2 = dhacw + dhacr;
        if (y2 < 0.95f * phwet || y2 == 0.f) {
            whacr = 0.0f; whaci = 0.0f; whacg = 0.0f; whacs = 0.0f;
        }
        primh = 0.0f;
        if (y2 > 0.f && phwet > 0.f && y2 < .95f * phwet) {
            float rim_frac = 2.0f * g4_min(g4_p2(1.f - G4DIV(y2, phwet)), 1.0f);
            rim_frac = rim_frac * g4_min(g4_p2(G4DIV(tairc, tdiff)), 1.0f);
            primh = rim_frac * dhacw;
            primh = g4_min(primh, G4DIV(qhv, d2t));
        }

        // Negative cloud ice (:3404-3453)
        y1 = G4DIV(qiv, d2t);
        psaut = g4_min(y1, psaut);
        psaci = g4_min(y1, psaci);
        praci = g4_min(y1, praci);
        psfi = g4_min(y1, psfi);
        dgaci = g4_min(y1, dgaci);
        wgaci = g4_min(y1, wgaci);
        whaci = g4_min(y1, whaci);
        qiv = qiv + d2t * (pihms + pihmg + pihmh);
        y1 = d2t * (psaut + psaci + praci + psfi + dgaci + wgaci + whaci);
        qiv = qiv - y1;
        if (qiv < 0.0f) {
            y2 = 1.f;
            if (y1 != 0.0f) y2 = G4DIV(qiv, y1) + 1.f;
            psaut = psaut * y2; psaci = psaci * y2; praci = praci * y2;
            psfi = psfi * y2; dgaci = dgaci * y2; wgaci = wgaci * y2;
            whaci = whaci * y2;
            qiv = 0.0f;
        }
        wgacr = qgacr + qgacw;
        dlt3 = 0.0f;
        if (qrv < 1.e-4f) dlt3 = 1.f;
        dlt4 = 1.f;
        if (qcv > 5.e-4f) dlt4 = 0.0f;
        if (qsv <= 1.e-4f) dlt4 = 1.f;
        if (tair >= c.t0) {
            dlt3 = 0.0f;
            dlt4 = 0.0f;
        }
        pr = d2t * (qsacw + praut + pracw + wgacr + qhacw - qgacr);
        ps = d2t * (psaut + psaci + dlt4 * psacw + psfw + psfi);
        pg = d2t * (dlt3 * praci + dgaci + wgaci + dgacw + (1.f - dlt4) * psacw + primh);

        // PRACS, PSACR, PGFR, PHFR (:3461-3505)
        qracs = 0.0f;
        y1 = fabsf(vr - vs);
        y2 = zr * zs;
        y3 = G4DIV(5.f, y2);
        y4 = .08f * y3 * y3;
        y5 = .05f * y3 * y4;
        pracs = r2ig * r2is * (r7rf * y1 * (G4DIV(y3, g4_p5(zs)) + G4DIV(y4, g4_p3(zs))
                                            + G4DIV(y5, zs)) * ftns * fros);
        if (qsv <= cmin) pracs = 0.f;
        if (qrv <= cmin) pracs = 0.f;
        qracs = r2ig * r2is * g4_min(d2t * pracs, qsv);
        psacr = r2is * (r8rf * y1 * (G4DIV(y3, g4_p5(zr)) + G4DIV(y4, g4_p3(zr))
                                     + G4DIV(y5, zr)) * ftns);
        if (qsv <= cmin) psacr = 0.f;
        if (qrv <= cmin) psacr = 0.f;
        qsacr = psacr;
        if (tair >= c.t0) {
            pracs = 0.0f;
            psacr = 0.0f;
        } else {
            qsacr = 0.0f;
            qracs = 0.0f;
        }
        pgfr = 0.0f;
        phfr = 0.0f;
        if (tair < c.t0) {
            y1 = g4_exp(c.rn18a * (c.t0 - tair));
            temp = G4DIV(1.f, zr);
            temp = temp * temp * temp * temp * temp * temp * temp;
            if (qrv >= 0.f) {
                phfr = r2ih * g4_max(r18r * (y1 - 1.f) * temp, 0.0f);
            } else {
                pgfr = r2ig * g4_max(r18r * (y1 - 1.f) * temp, 0.0f);
                if (qrv <= cmin) pgfr = 0.f;
            }
        }

        // Negative rain and snow; graupel and hail budgets (:3513-3603)
        y1 = G4DIV(qrv, d2t);
        piacr = g4_min(y1, piacr);
        dgacr = g4_min(y1, dgacr);
        dhacr = g4_min(y1, dhacr);
        psacr = g4_min(y1, psacr);
        pgfr = g4_min(y1, pgfr);
        phfr = g4_min(y1, phfr);
        del = 0.f;
        if (whacr < 0.f) del = 1.f;
        if (del == 1.f) whacr = g4_max(whacr, -dhacw);
        dhacr = g4_min((1.f - del) * whacr, dhacr);
        if (del == 1.f) dhacw = dhacw + del * whacr;
        y1 = (piacr + dgacr + dhacr + psacr + pgfr + phfr) * d2t;
        qrv = qrv + pr + qracs - del * whacr * d2t + qracg;
        qrv = qrv - y1;
        if (qrv < 0.0f) {
            y2 = 1.f;
            if (y1 != 0.0f) y2 = G4DIV(qrv, y1) + 1.f;
            piacr = piacr * y2; dgacr = dgacr * y2; dhacr = dhacr * y2;
            pgfr = pgfr * y2; phfr = phfr * y2; psacr = psacr * y2;
            qrv = 0.0f;
        }
        dlt2 = 1.f;
        if (qrv > 1.e-4f) dlt2 = 0.f;
        if (tair >= c.t0) dlt2 = 0.f;
        y1 = G4DIV(qsv, d2t);
        pgacs = g4_min(y1, pgacs);
        dgacs = g4_min(y1, dgacs);
        wgacs = g4_min(y1, wgacs);
        whacs = g4_min(y1, whacs);
        pracs = g4_min(y1, pracs);
        pwacs = g4_min(y1, pwacs);
        prn = d2t * (dlt3 * piacr + dlt2 * dgacr + pgfr + dlt2 * psacr);
        pracs = (1.f - dlt2) * pracs;
        pwacs = (1.f - dlt4) * pwacs;
        pracg = (1.f - dlt2) * pracg;
        psn = d2t * (pgacs + dgacs + wgacs + pracs + pwacs + whacs);
        qsv = qsv + ps - qracs - psn;
        if (qsv < 0.0f) {
            y2 = 1.f;
            if (psn != 0.f) y2 = G4DIV(qsv, psn) + 1.f;
            pgacs = pgacs * y2; dgacs = dgacs * y2; wgacs = wgacs * y2;
            whacs = whacs * y2; pracs = pracs * y2; pwacs = pwacs * y2;
            qsv = 0.0f;
        }
        psn = d2t * (pgacs + dgacs + wgacs + pracs + pwacs);
        qgv = qgv + pg + prn + psn - qracg;
        qgv = qgv - d2t * (pracs + whacg + pracg + pg2h);
        if (qgv < 0.0f) {
            y2 = 1.f;
            if (whacg + pracg + pg2h != 0.f)
                y2 = G4DIV(qgv, d2t * (whacg + pracg + pg2h)) + 1.f;
            whacg = whacg * y2;
            pracg = pracg * y2;
            pg2h = pg2h * y2;
            qgv = 0.0f;
        }
        qhv = qhv + d2t * (phfr + (1.f - dlt3) * piacr + (1.f - dlt3) * praci
                           + (1.f - dlt2) * psacr + pracs + (1.f - dlt2) * dgacr
                           + pracg + dhacw + dhacr - primh + whaci + whacs
                           + whacg + pg2h);
        y1 = d2t * (psacw + psfw + dgacw + piacr + dgacr + psacr + pgfr + phfr
                    + pihms + pihmg + dhacw + dhacr + pihmh) - qracs;
        pt = pt + afcp * y1;
        tair = (pt + tb0) * pi0;

        // PSMLT, PGMLT, PHMLT (:3608-3656)
        psmlt = 0.0f;
        pgmlt = 0.0f;
        phmlt = 0.0f;
        G4_SELECT_Z2();
        ftns0 = g4_sgmap(c, 1, qsv, qgv, qgz2, qhv, qhz2, r00, tairc);
        ftng0 = g4_sgmap(c, 2, qsv, qgv, qgz2, qhv, qhz2, r00, tairc);
        fros0 = g4_sgmap(c, 3, qsv, qgv, qgz2, qhv, qhz2, r00, tairc);
        ftnh0 = g4_sgmap(c, 4, qsv, qgv, qgz2, qhv, qhz2, r00, tairc);
        if (tair >= c.t0) {
            tairc = tair - c.t0;
            ftns = ftns0;
            ftng = ftng0;
            ftnh = ftnh0;
            dd = r11t * tairc * (G4DIV(r101r, g4_p2(zs)) + G4DIV(r102rf, g4_pow(zs, bsh5))) * ftns;
            psmlt = r2is * g4_min(qsv, g4_max(dd, 0.0f));
            if (r00 * qgv > c.qrog2)
                y2 = (G4DIV(r191r, g4_p2(zg)) + G4DIV(r192rf2, g4_pow(zg, bgh5_2))) * ftng;
            else
                y2 = (G4DIV(r191r, g4_p2(zg)) + G4DIV(r192rf, g4_pow(zg, bgh5))) * ftng;
            dd1 = tairc * (r19t * y2 + r19at * (qgacw + qgacr));
            pgmlt = r2ig * g4_min(qgv, g4_max(dd1, 0.0f));
            y1 = tca * tairc - alvr * dwv * (rp0 - (qvv + qb0));
            y3 = G4DIV(.78f, g4_p2(zh)) + G4DIV(h19aq * scv, g4_pow(zh, bhh5));
            dda0 = h19rt * y1 * y3 * ftnh + r19at * tairc * (qhacw + qhacr);
            phmlt = r2ih * g4_max(0.0f, g4_min(dda0, qhv));
            pt = pt - afcp * (psmlt + pgmlt + phmlt);
            tair = (pt + tb0) * pi0;
            qrv = qrv + psmlt + pgmlt + phmlt;
            qsv = qsv - psmlt;
            qgv = qgv - pgmlt;
            qhv = qhv - phmlt;
        }

        // PIHOM, PIDW, PIMLT, PIMM, PCFR (:3665-3823)
        if (qcv <= cmin1) qcv = 0.0f;
        if (qiv <= cmin1) qiv = 0.0f;
        // :3668-3680 recompute ftns0/ftng0 here; both are reassigned before
        // their next reader, so the two sgmap calls change no output word.
        ftns = 1.f;
        ftng = 1.f;
        ftns0 = 1.f;
        ftng0 = 1.f;
        G4_SELECT_Z2();
        if (tair <= c.t00) pihom = qcv;
        else pihom = 0.0f;
        if (tair >= c.t0) pimlt = qiv;
        else pimlt = 0.0f;
        pidw = 0.0f;
        pimm = 0.0f;
        pcfr = 0.0f;
        if (tair < c.t0 && tair > c.t00) {
            tairc = tair - c.t0;
            y1 = g4_max(g4_min(tairc, -1.f), -31.f);
            it = (int)fabsf(y1);
            y2 = c.aa1[it - 1];
            y3 = c.aa2[it - 1];
            if (tairc <= -5.f) {
                rtair = G4DIV(1.f, tair - c.c76);
                y5 = g4_exp(c.c218 - c.c580 * rtair);
                qsi = rp0 * y5;
                ssi = G4DIV(qvv + qb0, qsi) - 1.f;
                fssi = rssi;
                wssi = G4DIV(ww1, 100.f) - 2.0f;
                if (wssi > 0.f) fssi = rssi + g4_min(xssi, wssi * 0.01f);
                fssi = g4_min(ssi, fssi);
                if (tairc < -40.0f) c_nci = c_nci_cap;
                else c_nci = 5.0e-6f * g4_exp(0.304f * fabsf(tairc));
                r_nci = c_nci;
                r_nci = g4_max(r_nci, G4DIV(r00 * qiv, c.ami50));
                dd = g4_pow(G4DIV(r00 * qiv, r_nci), y3);
                esi = G4DIV(qsi, rp0) * c.c610;
                y4 = G4DIV(1.f, tair);
                term1 = y4 * (c.rn20a * y4 - c.rn20b);
                term2 = G4DIV(c.rn10c * tair, esi);
                fdwv = G4DIV(term1 + term2, term1 + G4DIV(term2 * dwv0, dwv));
                pidw = g4_min(rr0 * d2t * y2 * r_nci * dd * fdwv, qcv);
            }
            pimm = 0.0f;
            pcfr = 0.0f;
            if (qcv > 0.0f) {
                y4 = G4DIV(1.f, tair - c.c358);
                qsw = rp0 * g4_exp(c.c172 - c.c409 * y4);
                const float xncld = G4DIV(qcv, 4.e-9f);
                const float esat = 0.6112f * g4_exp(G4DIV(17.67f * tairc, tairc + 243.5f)) * 10.f;
                const float rv = G4DIV(0.622f * esat, G4DIV(p0, 1000.f) - esat);
                const float rlapse_m = G4DIV(
                    980.616f * (1.f + G4DIV(G4DIV(2.5e6f * rv, 287.f), tair)),
                    1004.67f + G4DIV(6.25e12f * rv * 0.622f, 287.f * tair * tair));
                float delT = rlapse_m * ww1;
                if (delT < 0.f) delT = 0.f;
                const float Bhi = 1.01e-2f;
                pimm = xncld * Bhi * 4.e-9f * g4_exp(-tairc) * delT * d2t * 4.e-9f;
                const float xccld = xncld * r00;
                const float Xknud = G4DIV(7.37f * tair, 288.f * Ra * p0);
                const float alpha = 1.257f + 0.400f * g4_exp(G4DIV(-1.10f, Xknud));
                const float cunnF = 1.f + alpha * Xknud;
                float dvair;
                if (tairc >= 0.f) dvair = (1.718f + 0.0049f * tairc) * 1.e-4f;
                else dvair = (1.718f + 0.0049f * tairc - 1.2e-5f * g4_p2(tairc)) * 1.e-4f;
                const float DIFFar = G4DIV(G4DIV(G4DIV(G4DIV(1.3804e-16f * tair, 6.f), c.cpi),
                                                 dvair), Ra) * cunnF;
                if (qvv + qb0 - qsw < 0.f)
                    pcfr = 4.e-9f * 4.f * c.cpi * Rc * DIFFar * xccld * Cna * rr0 * d2t;
            } else {
                tairc = tair - c.t0;
                y1 = g4_max(g4_min(tairc, -1.f), -31.f);
                it = (int)fabsf(y1);
                y2 = c.aa1[it - 1];
                y3 = c.aa2[it - 1];
                y4 = g4_exp(fabsf(.5f * tairc));
                dd = g4_pow(G4DIV(r00 * qiv, r25a * y4), y3);
                pidw = g4_min(r25rt * y2 * y4 * dd, qcv);
            }
        }
        y1 = pihom + pidw + pimm + pcfr - pimlt;
        if (y1 > qcv) {
            y1 = qcv;
            y2 = 1.f;
            y3 = pihom + pidw + pimm + pcfr;
            if (y3 != 0.f) y2 = G4DIV(qcv + pimlt, y3);
            pihom = pihom * y2;
            pidw = pidw * y2;
            pimm = pimm * y2;
            pcfr = pcfr * y2;
        }
        pt = pt + afcp * y1;
        tair = (pt + tb0) * pi0;
        qcv = qcv - y1;
        qiv = qiv + y1;

        // PINT, PIDEP (:3832-3906)
        tair = (pt + tb0) * pi0;
        if (tair < c.t0) {
            if (qiv <= cmin) qiv = 0.f;
            tairc = tair - c.t0;
            rtair = G4DIV(1.f, tair - c.c76);
            y2 = g4_exp(c.c218 - c.c580 * rtair);
            qsi = rp0 * y2;
            esi = c.c610 * y2;
            ssi = G4DIV(qvv + qb0, qsi) - 1.f;
            y1 = G4DIV(1.f, tair);
            y3 = G4SQRT(qiv);
            term1 = y1 * (c.rn20a * y1 - c.rn20b);
            term2 = G4DIV(c.rn10c * tair, esi);
            dd = term1 + term2;
            fdwv = G4DIV(dd, term1 + G4DIV(term2 * dwv0, dwv));
            dm = g4_max(qvv + qb0 - qsi, 0.0f);
            rsub1 = cs580 * qsi * rtair * rtair;
            dep = G4DIV(dm, 1.f + rsub1);
            if (tairc <= -5.f) {
                fssi = rssi;
                wssi = G4DIV(ww1, 100.f) - 2.0f;
                if (wssi > 0.f) fssi = rssi + g4_min(xssi, wssi * 0.01f);
                fssi = g4_min(ssi, fssi);
                if (tairc < -40.0f) c_nci = c_nci_cap;
                else c_nci = 5.0e-6f * g4_exp(0.304f * fabsf(tairc));
                r_nci = c_nci;
                r_nci = g4_max(r_nci, G4DIV(r00 * qiv, c.ami50));
                pidep = g4_max(G4DIV(r32rt * 1.e4f * fssi * G4SQRT(r_nci) * y3, dd) * fdwv,
                               -qiv);
                if (qiv <= cmin) pidep = 0.f;
                dd = g4_max(G4DIV(1.e-9f * r_nci, r00) - G4DIV(qi_in * 1.e-9f, c.ami50), 0.f);
                pint = g4_max(g4_min(dd, dm), 0.f);
                pint = g4_min(pint + pidep, dep);
                pt = pt + ascp * pint;
                tair = (pt + tb0) * pi0;
                qvv = qvv - pint;
                qiv = qiv + pint;
            }
        }

        // Saturation adjustment, new_ice_sat = 9 (:3922-3971)
        dep = 0.0f;
        cnd = 0.0f;
        tair = (pt + tb0) * pi0;
        if (tair >= c.t00) {
            y1 = G4DIV(1.f, tair - c.c358);
            qsw = rp0 * g4_exp(c.c172 - c.c409 * y1);
            dd = cp409 * y1 * y1;
            dm = qvv + qb0 - qsw;
            cnd = G4DIV(dm, 1.f + avcp * dd * qsw);
            cnd = g4_max(-qcv, cnd);
            if (ww1 >= thresh_evap) cnd = g4_max(0.f, cnd);
            pt = pt + avcp * cnd;
            tair = (pt + tb0) * pi0;
            qvv = qvv - cnd;
            qcv = qcv + cnd;
        }
        if (tair <= 273.16f) {
            y1 = G4DIV(1.f, tair - c.c358);
            qsw = rp0 * g4_exp(c.c172 - c.c409 * y1);
            y2 = G4DIV(1.f, tair - c.c76);
            qsi = rp0 * g4_exp(c.c218 - c.c580 * y2);
            fssi = rssi;
            wssi = G4DIV(ww1, 100.f) - 2.0f;
            if (wssi > 0.f) fssi = rssi + g4_min(xssi, wssi * 0.01f);
            y3 = 1.f + g4_min(G4DIV(qsw - qsi, qsi), fssi);
            y4 = qsi * y3;
            if (tair <= 268.16f && (qvv + qb0 > y4)) {
                dd1 = cp580 * y2 * y2;
                dep = G4DIV(qvv + qb0 - y4, 1.f + ascp * dd1 * y4);
            } else if (qvv + qb0 < xsubi * qsi && qiv > cmin) {
                dd1 = cp580 * y2 * y2;
                dep = G4DIV(qvv + qb0 - xsubi * qsi, 1.f + ascp * dd1 * xsubi * qsi);
                dep = g4_max(-qiv, dep);
            }
            if (ww1 >= 0.f) dep = g4_max(0.f, dep);
            pt = pt + ascp * dep;
            tair = (pt + tb0) * pi0;
            qvv = qvv - dep;
            qiv = qiv + dep;
        }

        // PSDEP, PGDEP, PHDEP and sublimation (:3976-4103)
        psdep = 0.0f; pgdep = 0.0f; phdep = 0.0f; pssub = 0.0f;
        pgsub = 0.0f; phsub = 0.0f; pvapg = 0.0f; pvaph = 0.0f;
        if (qcv + qiv > 1.e-5f) dlt1 = 1.f;
        else dlt1 = 0.f;
        if (tair < c.t0) {
            rtair = G4DIV(1.f, tair - c.c76);
            y2 = g4_exp(c.c218 - c.c580 * rtair);
            qsi = rp0 * y2;
            esi = c.c610 * y2;
            ssi = G4DIV(qvv + qb0, qsi) - 1.f;
            if (ssi > 0.f) dlt1 = 1.f;
            if (ssi <= 0.f) dlt1 = 0.f;
            dm = qvv + qb0 - qsi;
            rsub1 = cs580 * qsi * rtair * rtair;
            dd1 = G4DIV(dm, 1.f + rsub1);
            y3 = G4DIV(1.f, tair);
            term1 = y3 * (c.rn20a * y3 - c.rn20b);
            term2 = G4DIV(c.rn10c * tair, esi);
            dd = term1 + term2;
            fdwv = G4DIV(dd, term1 + G4DIV(term2 * dwv0, dwv));
            tairc = tair - c.t0;
            G4_SELECT_Z2();
            ftns0 = g4_sgmap(c, 1, qsv, qgv, qgz2, qhv, qhz2, r00, tairc);
            ftng0 = g4_sgmap(c, 2, qsv, qgv, qgz2, qhv, qhz2, r00, tairc);
            fros0 = g4_sgmap(c, 3, qsv, qgv, qgz2, qhv, qhz2, r00, tairc);
            ftnh0 = g4_sgmap(c, 4, qsv, qgv, qgz2, qhv, qhz2, r00, tairc);
            ftns = ftns0;
            ftng = ftng0;
            ftnh = ftnh0;
            y4 = G4DIV(r10t * ssi * (G4DIV(r101r, g4_p2(zs)) + G4DIV(r102rf, g4_pow(zs, bsh5))),
                       dd) * ftns * fdwv;
            psdep = r2is * g4_max(-qsv, y4);
            if (qsv <= cmin) psdep = 0.f;
            dd = y3 * (c.rn20a * y3 - c.rn20b) + G4DIV(c.rn10c * tair, esi);
            y2 = G4DIV(r191r, g4_p2(zg)) + G4DIV(r192rf, g4_pow(zg, bgh5));
            if (r00 * qgv > c.qrog2)
                y2 = (G4DIV(r191r, g4_p2(zg)) + G4DIV(r192rf2, g4_pow(zg, bgh5_2)));
            pgdep = r2ig * g4_max(-qgv, G4DIV(r20t * ssi * y2, dd) * ftng * fdwv);
            if (qgv <= cmin) pgdep = 0.f;
            y1 = G4DIV(h10ar, tca * g4_p2(tair)) + G4DIV(1.f, dwv * qsi);
            y2 = G4DIV(.78f, g4_p2(zh)) + G4DIV(h20bq * scv, g4_pow(zh, bhh5));
            phdep = r2ih * g4_max(-qhv, G4DIV(h20t * ftnh * ssi * y2, y1));
            if (qhv <= cmin) phdep = 0.f;
            pssub = g4_min(psdep, 0.f);
            psdep = g4_max(psdep, 0.f);
            pgsub = g4_min(pgdep, 0.f);
            pgdep = g4_max(pgdep, 0.f);
            phsub = g4_min(phdep, 0.f);
            phdep = g4_max(phdep, 0.f);
            y5 = g4_min(0.f, dd1);
            dd1 = g4_max(0.f, dd1);
            if (dlt1 == 1.f) {
                y1 = psdep + pgdep + phdep;
                if (y1 >= dd1) {
                    psdep = G4DIV(psdep, y1) * dd1;
                    pgdep = G4DIV(pgdep, y1) * dd1;
                    phdep = G4DIV(phdep, y1) * dd1;
                }
            }
            if (qcv <= 1.e-5f) {
                const float vap_frac = 2.0f * g4_min(g4_p2(G4DIV(tairc, tdiff)), 1.0f);
                pvapg = vap_frac * pgdep;
                pvaph = vap_frac * phdep;
            }
            if (pgdep > 0.f) pvapg = g4_min(pvapg, qgv + pgdep);
            if (phdep > 0.f) pvaph = g4_min(pvaph, qhv + phdep);
            if (dlt1 == 0.f) {
                y1 = g4_max(pssub + pgsub + phsub, y5);
                if (y5 > (pssub + pgsub + phsub)) {
                    y3 = (pssub + pgsub + phsub);
                    if (y3 != 0.0f) {
                        pssub = G4DIV(pssub, y3) * y5;
                        pgsub = G4DIV(pgsub, y3) * y5;
                        phsub = G4DIV(phsub, y3) * y5;
                    }
                }
            }
            pssub = -pssub;
            pgsub = -pgsub;
            phsub = -phsub;
            y1 = psdep + pgdep + phdep - pssub - pgsub - phsub;
            pt = pt + ascp * y1;
            tair = (pt + tb0) * pi0;
            qvv = qvv - y1;
            qsv = qsv + psdep - pssub + pvapg + pvaph;
            qgv = qgv + pgdep - pgsub - pvapg;
            qhv = qhv + phdep - phsub - pvaph;
        }

        // ERN (:4107-4145)
        ern = 0.0f;
        if (qrv > 0.0f) {
            tair = (pt + tb0) * pi0;
            rtair = G4DIV(1.f, tair - c.c358);
            y2 = g4_exp(c.c172 - c.c409 * rtair);
            esw = c.c610 * y2;
            qsw = rp0 * y2;
            ssw = G4DIV(qvv + qb0, qsw) - 1.f;
            dm = qvv + qb0 - qsw;
            rsub1 = cv409 * qsw * rtair * rtair;
            dd1 = g4_max(-G4DIV(dm, 1.f + rsub1), 0.0f);
            y3 = G4DIV(1.f, tair);
            term1 = y3 * (c.rn30a * y3 - c.rn10b);
            term2 = G4DIV(c.rn10c * tair, esw);
            dd = term1 + term2;
            fdwv = G4DIV(dd, term1 + G4DIV(term2 * dwv0, dwv));
            ftnw = 1.f;
            if (qrv > cmin && tair > c.t0) {
                bin_factor = 0.11f * g4_pow(1000.f * qrv, -1.27f) + 0.98f;
                bin_factor = g4_min(bin_factor, 1.30f);
                ftnw = G4DIV(1.f, g4_pow(bin_factor, 3.35f));
                ftnwmin = G4DIV(r00 * qrv, c.draimax);
                if (qrv <= 0.001f) ftnw = g4_max(ftnw, G4DIV(ftnwmin, c.tnw));
                y4 = r00 * qrv;
                y1 = G4SQRT(y4);
                y2 = G4SQRT(y1);
                zr = G4DIV(c.zrc, y2) * g4_pow(ftnw, 0.25f);
            }
            y1 = -(G4DIV(r23t * ssw * (G4DIV(r231r, g4_p2(zr)) + G4DIV(r232rf, g4_p3(zr))), dd)
                   * ftnw * fdwv);
            ern = g4_min(g4_min(dd1, qrv), g4_max(y1, 0.0f));
            pt = pt - avcp * ern;
            tair = (pt + tb0) * pi0;
            tairc = tair - c.t0;
            qvv = qvv + ern;
            qrv = qrv - ern;
        }

        // PMLTG, PMLTS (:4155-4209)
        pmlts = 0.0f;
        pmltg = 0.0f;
        tair = (pt + tb0) * pi0;
        tairc = tair - c.t0;
        ftns0 = 1.f;
        ftng0 = 1.f;
        G4_SELECT_Z2();
        ftns0 = g4_sgmap(c, 1, qsv, qgv, qgz2, qhv, qhz2, r00, tairc);
        ftng0 = g4_sgmap(c, 2, qsv, qgv, qgz2, qhv, qhz2, r00, tairc);
        fros0 = g4_sgmap(c, 3, qsv, qgv, qgz2, qhv, qhz2, r00, tairc);
        if (tair >= c.t0) {
            ftns = ftns0;
            ftng = ftng0;
            rtair = G4DIV(1.f, c.t0 - c.c358);
            y2 = g4_exp(c.c172 - c.c409 * rtair);
            esw = c.c610 * y2;
            qsw = rp0 * y2;
            ssw = 1.f - G4DIV(qvv + qb0, qsw);
            dm = qsw - qvv - qb0;
            rsub1 = cv409 * qsw * rtair * rtair;
            dd1 = g4_max(G4DIV(dm, 1.f + rsub1), 0.0f);
            y3 = G4DIV(1.f, tair);
            term1 = y3 * (c.rn30a * y3 - c.rn10b);
            term2 = G4DIV(c.rn10c * tair, esw);
            dd = term1 + term2;
            fdwv = G4DIV(dd, term1 + G4DIV(term2 * dwv0, dwv));
            y1 = G4DIV(ftng * r30t * ssw * (G4DIV(r191r, g4_p2(zg)) + G4DIV(r192rf, g4_pow(zg, bgh5))),
                       dd) * fdwv;
            if (r00 * qgv > c.qrog2)
                y1 = G4DIV(ftng * r30t * ssw * (G4DIV(r191r, g4_p2(zg))
                                                + G4DIV(r192rf2, g4_pow(zg, bgh5_2))), dd) * fdwv;
            pmltg = r2ig * g4_min(qgv, g4_max(y1, 0.0f));
            y1 = G4DIV(ftns * r33t * ssw * (G4DIV(r331r, g4_p2(zs)) + G4DIV(r332rf, g4_pow(zs, bsh5))),
                       dd) * fdwv;
            pmlts = r2is * g4_min(qsv, g4_max(y1, 0.0f));
            y1 = g4_min(pmltg + pmlts, dd1);
            pmltg = y1 - pmlts;
            pt = pt - ascp * y1;
            tair = (pt + tb0) * pi0;
            qvv = qvv + y1;
            qsv = qsv - pmlts;
            qgv = qgv - pmltg;
        }

        if (qcv <= cmin) qcv = 0.f;
        if (qrv <= cmin) qrv = 0.f;
        if (qiv <= cmin) qiv = 0.f;
        if (qsv <= cmin) qsv = 0.f;
        if (qgv <= cmin) qgv = 0.f;
        if (qhv <= cmin) qhv = 0.f;

        // Latent heating diagnostics (:4231-4264)
        dd = g4_max(-cnd, 0.f);
        cnd = g4_max(cnd, 0.f);
        dd1 = g4_max(-dep, 0.f);
        dep = g4_max(dep, 0.f);
        del = 0.f;
        if (whacr < 0.f) del = 1.f;
        const float sccc = cnd;
        const float seee = dd + ern;
        const float sddd = dep + g4_max(pint, 0.0f) + psdep + pgdep + phdep;
        const float ssss = dd1 - g4_min(pint, 0.0f) + pssub + pgsub + phsub + pmlts + pmltg;
        const float smmm = psmlt + pgmlt + pimlt + qracs + phmlt + qracg - del * whacr;
        const float sfff = psacw * d2t + piacr * d2t + psfw * d2t + pgfr * d2t
                         + dgacw * d2t + dgacr * d2t + psacr * d2t + pihom
                         + pidw + pimm + pcfr + pihms * d2t
                         + pihmg * d2t + phfr * d2t + dhacw * d2t + dhacr * d2t
                         + pihmh * d2t;
        G4AT(physc, k) = G4DIV(c.avc * sccc, d2t);
        G4AT(physe, k) = G4DIV(c.avc * seee, d2t);
        G4AT(physd, k) = G4DIV(c.asc * sddd, d2t);
        G4AT(physs, k) = G4DIV(c.asc * ssss, d2t);
        G4AT(physf, k) = G4DIV(c.afc * sfff, d2t);
        G4AT(physm, k) = G4DIV(c.afc * smmm, d2t);
        G4AT(acphysc, k) = (zero_acc ? 0.f : G4AT(acphysc, k)) + avcp * sccc;
        G4AT(acphyse, k) = (zero_acc ? 0.f : G4AT(acphyse, k)) + avcp * seee;
        G4AT(acphysd, k) = (zero_acc ? 0.f : G4AT(acphysd, k)) + ascp * sddd;
        G4AT(acphyss, k) = (zero_acc ? 0.f : G4AT(acphyss, k)) + ascp * ssss;
        G4AT(acphysf, k) = (zero_acc ? 0.f : G4AT(acphysf, k)) + afcp * sfff;
        G4AT(acphysm, k) = (zero_acc ? 0.f : G4AT(acphysm, k)) + afcp * smmm;

        // Reflectivity (:4271-4318)
        ftns0 = g4_sgmap(c, 1, qsv, qgv, qgz2, qhv, qhz2, r00, tairc);
        ftng0 = g4_sgmap(c, 2, qsv, qgv, qgz2, qhv, qhz2, r00, tairc);
        fros0 = g4_sgmap(c, 3, qsv, qgv, qgz2, qhv, qhz2, r00, tairc);
        ftnh0 = g4_sgmap(c, 4, qsv, qgv, qgz2, qhv, qhz2, r00, tairc);
        const float a_1 = 1.e6f * r00 * qrv;
        const float a_2 = 1.e6f * r00 * qsv;
        const float a_3 = 1.e6f * r00 * qgv;
        const float a_4 = 1.e6f * r00 * qhv;
        ftnw = 1.f;
        if (qrv > cmin && qcv < cmin) {
            bin_factor = 0.11f * g4_pow(1000.f * qrv, -1.27f) + 0.98f;
            bin_factor = g4_min(bin_factor, 1.30f);
            ftnw = G4DIV(1.f, g4_pow(bin_factor, 3.50f));
            ftnwmin = G4DIV(r00 * qrv, c.draimax);
            if (qrv <= 0.001f) ftnw = g4_max(ftnw, G4DIV(ftnwmin, c.tnw));
        }
        const float a_11 = G4DIV(c.ucor * g4_pow(g4_max(0.f, a_1), 1.75f), g4_pow(ftnw, 0.75f));
        ftns = g4_pow(ftns0, 0.75f);
        ftng = g4_pow(ftng0, 0.75f);
        ftnh = g4_pow(ftnh0, 0.75f);
        fros = g4_pow(fros0, .25f);
        const float a_22 = G4DIV(c.ucos * g4_pow(g4_max(0.f, a_2), 1.75f), ftns) * fros;
        float a_33 = G4DIV(c.ucog * g4_pow(g4_max(0.f, a_3), 1.75f), ftng);
        if (a_3 >= c.qrog2) a_33 = G4DIV(c.ucog2 * g4_pow(g4_max(0.f, a_3), 1.75f), ftng);
        const float a_44 = G4DIV(c.ucoh * g4_pow(g4_max(0.f, a_4), 1.75f), ftnh);
        float dbz;
        if (tair < 273.16f) {
            const float zdry = g4_max(1.e-9f, a_11 + a_22 + a_33 + a_44);
            dbz = 10.f * g4_log10(zdry);
        } else {
            const float zwet0 = a_11 + c.uwet * g4_pow(a_22 + a_33 + a_44, .95f);
            const float zwet = g4_max(1.e-9f, zwet0);
            dbz = 10.f * g4_log10(zwet);
        }
        if (refl_due) G4AT(refl_10cm, k) = g4_max(-35.f, dbz);

        // Effective radii (:4343-4464)
        G4AT(re_rain, k) = (qrv < cmin) ? 0.e0f : g4_eff_rad(zr);
        G4AT(re_snow, k) = (qsv < cmin) ? 0.e0f : g4_eff_rad(zs);
        G4AT(re_graupel, k) = (qgv < cmin) ? 0.e0f : g4_eff_rad(zg);
        G4AT(re_hail, k) = (qhv < cmin) ? 0.e0f : g4_eff_rad(zh);
        if (qcv < cmin) {
            G4AT(re_cloud, k) = 0.e0f;
        } else {
            const float l_cloud = qcv * rhok;
            const float ccn_ref = land ? 1500.f : 150.f;
            const float gamfac1 = land ? c.gamfac1_land : c.gamfac1_water;
            const float gamfac3 = land ? c.gamfac3_land : c.gamfac3_water;
            const float lambda = g4_pow(G4DIV(g4_bits(0x3faaaaabu) * c.cpi * c.roqr * ccn_ref,
                                              l_cloud) * gamfac1, g4_bits(0x3eaaaaabu));
            G4AT(re_cloud, k) = G4DIV(1.e0f, lambda) * gamfac3 * 1.e4f;
        }
        if (qiv < cmin) {
            G4AT(re_ice, k) = 0.e0f;
        } else {
            float rei = 125.e0f + (tair - 243.16f) * 5.e0f;
            if (tair > 243.16f) rei = 125.e0f;
            if (tair < 223.16f) rei = 25.e0f;
            G4AT(re_ice, k) = rei;
        }

        G4AT(th, k) = pt;
        G4AT(qv, k) = qvv;
        G4AT(ql, k) = qcv;
        G4AT(qr, k) = qrv;
        G4AT(qi, k) = qiv;
        G4AT(qs, k) = qsv;
        G4AT(qg, k) = qgv;
        G4AT(qh, k) = qhv;
    }
#undef G4_SELECT_Z2
}
