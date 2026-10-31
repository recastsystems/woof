// Legacy RRTMG forecast adapter glue on the device (gpuwm/core/rrtmg_legacy.py).
//
// Each kernel is a statement-for-statement transcription of the NumPy
// float32 expression it replaces in the adapter, so the device-resident
// adapter returns the same bits the host path returned:
//
//   rla_gather     the chunk's column blocks of every input grid, one launch
//   rla_t8w        _t8w_columns (phy_prep interface temperatures)
//   rla_scale      legacy_radius_meters (x * 1e-6)
//   rla_mynn       mynn_bl_cloud_supplied + merge_mynn_bl_clouds +
//                  unsized_mynn_radii, in the adapter's order
//   rla_ozn_p_int  gpuwm.ingest.wrf_ozone.ozn_p_int
//   rla_lw_out     lwrad_outputs_batch's rthratenlw/glw/olr, per chunk in
//                  the (nz, nc) grid order the driver consumes
//   rla_sw_out     the adapter's day-column gsw/rthratensw lines
//
// Numerics: NumPy float32 add/sub/mul/div are single correctly rounded IEEE
// operations with gradual underflow, and these inputs are mixing ratios and
// radii whose values can be FP32 subnormals.  So the arithmetic is inline
// PTX without the .ftz modifier (add.rn.f32, sub.rn.f32, mul.rn.f32,
// div.rn.f32), which no compile option can turn into a flushing
// instruction (the rrtmg_sw.cu header documents the same idiom and the
// receipt behind it), and the unit is compiled through NVRTC directly with
// --ftz=false (rrtmg_legacy._adapter_module) so comparisons keep
// subnormals too.  rla_probe proves both on the live device before first
// use.  No transcendental appears here.

#define RLA_AD(a, b) rla_add((a), (b))
#define RLA_SU(a, b) rla_sub((a), (b))
#define RLA_MU(a, b) rla_mul((a), (b))
#define RLA_DV(a, b) rla_div((a), (b))

static __device__ __forceinline__ float rla_add(float a, float b)
{ float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
static __device__ __forceinline__ float rla_sub(float a, float b)
{ float r; asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
static __device__ __forceinline__ float rla_mul(float a, float b)
{ float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
static __device__ __forceinline__ float rla_div(float a, float b)
{ float r; asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }

// Live-device proof: o[0] = x0 * x1 (a subnormal product must survive),
// o[1] = (x2 > 0) for a subnormal x2 (a compare must not flush it),
// o[2] = x3 / x4 (a subnormal quotient must survive).
extern "C" __global__ void rla_probe(const float* x, float* o)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    o[0] = RLA_MU(x[0], x[1]);
    o[1] = (x[2] > 0.0f) ? 1.0f : 0.0f;
    o[2] = RLA_DV(x[3], x[4]);
}

// The chunk's column blocks of several (nk_f, ncol) grids in one launch:
// block f is (nc, nk_f), row r holding column idx[r] (or c0 + r when idx is
// null), packed back to back in buf in field order.  Data movement only.
extern "C" __global__ void rla_gather(
    int nfield, int nc, long long ncol, long long c0,
    const long long* __restrict__ idx,
    const unsigned long long* __restrict__ src,   // [nfield] float* grids
    const int* __restrict__ nk,                   // [nfield]
    float* __restrict__ buf)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long base = 0;
    int f = 0;
    for (; f < nfield; ++f) {
        long long n = (long long)nc * nk[f];
        if (i < base + n) break;
        base += n;
    }
    if (f >= nfield) return;
    long long local = i - base;
    int levels = nk[f];
    int k = (int)(local % levels);
    long long row = local / levels;
    long long col = (idx != nullptr) ? idx[row] : c0 + row;
    buf[i] = ((const float*)src[f])[(long long)k * ncol + col];
}

// t8w (ncol, nz+1) from t3d (ncol, nz), z_at_w (ncol, nz+1) and the
// fnm/fnp weights (nz,).  One thread per output (column, interface).
extern "C" __global__ void rla_t8w(
    int ncol, int nz,
    const float* __restrict__ t3d, const float* __restrict__ zw,
    const float* __restrict__ fnm, const float* __restrict__ fnp,
    float* __restrict__ t8w)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int n1 = nz + 1;
    if (i >= (long long)ncol * n1) return;
    int col = (int)(i / n1);
    int k = (int)(i % n1);
    const float* t = t3d + (long long)col * nz;
    const float* w = zw + (long long)col * n1;
    float out;
    if (k >= 1 && k <= nz - 1) {
        out = RLA_AD(RLA_MU(fnm[k], t[k]), RLA_MU(fnp[k], t[k - 1]));
    } else if (k == 0) {
        float z0 = RLA_MU(0.5f, RLA_AD(w[0], w[1]));
        float z1 = RLA_MU(0.5f, RLA_AD(w[1], w[2]));
        float w1 = RLA_DV(RLA_SU(w[0], z1), RLA_SU(z0, z1));
        float w2 = RLA_SU(1.0f, w1);
        out = RLA_AD(RLA_MU(w1, t[0]), RLA_MU(w2, t[1]));
    } else {
        float zm2 = RLA_MU(0.5f, RLA_AD(w[nz - 2], w[nz - 1]));
        float zm1 = RLA_MU(0.5f, RLA_AD(w[nz - 1], w[nz]));
        float w1 = RLA_DV(RLA_SU(w[nz], zm2), RLA_SU(zm1, zm2));
        float w2 = RLA_SU(1.0f, w1);
        out = RLA_AD(RLA_MU(w1, t[nz - 1]), RLA_MU(w2, t[nz - 2]));
    }
    t8w[i] = out;
}

// out = x * s elementwise (s is the float32 1e-6 of legacy_radius_meters).
extern "C" __global__ void rla_scale(
    long long n, const float* __restrict__ x, float s, float* __restrict__ out)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = RLA_MU(x[i], s);
}

// The MYNN subgrid-cloud merge for radiation, in the adapter's order:
// supplied masks from the PRE-merge qc/qi, then the fraction replacement
// (after the first timestep), then the mass merge, then the unsized radii.
// cldfra, re_cloud and re_ice may be null (not present); every array is
// (n,) in the same column-major-free flat order.
extern "C" __global__ void rla_mynn(
    long long n,
    float* __restrict__ qc, float* __restrict__ qi,
    const float* __restrict__ qc_bl, const float* __restrict__ qi_bl,
    const float* __restrict__ cldfra_bl,
    float* __restrict__ cldfra, int replace_cldfra,
    float* __restrict__ re_cloud, float* __restrict__ re_ice, int ice_rule,
    float qc_below, float qi_below, float cf_above)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float c = qc[i];
    float e = qi[i];
    float cb = qc_bl[i];
    float ib = qi_bl[i];
    float fb = cldfra_bl[i];
    bool cloudy = fb > cf_above;
    bool liq_low = c < qc_below;
    bool ice_low = e < qi_below;
    bool supplied_liquid = liq_low && cloudy && (cb > 0.0f);
    bool supplied_ice = ice_low && cloudy && (ib > 0.0f);
    if (cldfra != nullptr && replace_cldfra) cldfra[i] = fb;
    qc[i] = (liq_low && cloudy) ? RLA_AD(c, cb) : c;
    qi[i] = (ice_low && cloudy) ? RLA_AD(e, ib) : e;
    if (re_cloud != nullptr && supplied_liquid) re_cloud[i] = 0.0f;
    if (re_ice != nullptr && ice_rule && supplied_ice) re_ice[i] = 0.0f;
}

// The GSD MYNN v4.1 subgrid-cloud merge (bl_mynn_version = "gsd_41"),
// NOAA-EMC WRF 3.9 module_radiation_driver.F:1256-1303.  On the first model
// step the fraction takes CLDFRA_BL only where it exceeds 0.001; after it,
// everywhere.  In-cloud QC_BL times CLDFRA_BL is added only where neither
// resolved phase is present (qc < 1e-6 and qi < 1e-8) and CLDFRA_BL >
// 0.001, split liquid/ice by MIN(1, MAX(0, (T-254)/15)).  The radii of the
// layers it supplies are left unsized, as rla_mynn does.
extern "C" __global__ void rla_mynn_gsd41(
    long long n,
    float* __restrict__ qc, float* __restrict__ qi,
    const float* __restrict__ qc_bl, const float* __restrict__ cldfra_bl,
    const float* __restrict__ t,
    float* __restrict__ cldfra, int first_step,
    float* __restrict__ re_cloud, float* __restrict__ re_ice, int ice_rule,
    float qc_below, float qi_below, float cf_above)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float c = qc[i];
    float e = qi[i];
    float cb = qc_bl[i];
    float fb = cldfra_bl[i];
    if (cldfra != nullptr) {
        if (!first_step || fb > cf_above) cldfra[i] = fb;
    }
    if (c < qc_below && e < qi_below && fb > cf_above) {
        float x = RLA_DV(RLA_SU(t[i], 254.0f), 15.0f);
        float liq = x > 0.0f ? x : 0.0f;
        liq = liq < 1.0f ? liq : 1.0f;
        float add_l = RLA_MU(RLA_MU(cb, liq), fb);
        float add_i = RLA_MU(RLA_MU(cb, RLA_SU(1.0f, liq)), fb);
        qc[i] = RLA_AD(c, add_l);
        qi[i] = RLA_AD(e, add_i);
        if (re_cloud != nullptr && add_l > 0.0f) re_cloud[i] = 0.0f;
        if (re_ice != nullptr && ice_rule && add_i > 0.0f) re_ice[i] = 0.0f;
    }
}

// ozn_p_int: one thread per column, the same top-down walk with the
// carried kupper and the three branches (above the climatology top, below
// its bottom, bracketed or pm == pin(1) with the stale kupper).  Element
// (k, col) of p and o3 is at k * ks + col * cs, so one kernel serves the
// (pver, ncol) grid layout (ks = ncol, cs = 1) and the (ncol, pver) column
// block (ks = 1, cs = pver).
extern "C" __global__ void rla_ozn_p_int(
    int ncol, int pver, int levsiz,
    const float* __restrict__ p, long long pks, long long pcs,
    const float* __restrict__ pin,      // (levsiz,) increasing
    const float* __restrict__ ozmixt,   // (ncol, levsiz)
    float* __restrict__ o3, long long oks, long long ocs)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const float* oz = ozmixt + (long long)col * levsiz;
    int kupper = 0;
    float pin0 = pin[0];
    float pinl = pin[levsiz - 1];
    for (int k = 0; k < pver; ++k) {
        int kout = pver - 1 - k;
        // pmid = p[:, ::-1]; pm = pmid[:, k]
        float pm = p[(long long)kout * pks + (long long)col * pcs];
        // searchsorted(pin, pm, side="left") - 1: the count of pin < pm, less one.
        int cnt = 0;
        for (int j = 0; j < levsiz; ++j) cnt += (pin[j] < pm) ? 1 : 0;
        int cand = cnt - 1;
        bool found = (cand >= 0) && (cand <= levsiz - 2);
        if (found) kupper = cand;
        bool above = !found && (pm < pin0);
        bool below = !found && (pm > pinl);
        float res;
        if (above) {
            res = RLA_DV(RLA_MU(oz[0], pm), pin0);
        } else if (below) {
            res = oz[levsiz - 1];
        } else {
            int ku = kupper;
            float dpu = RLA_SU(pm, pin[ku]);
            float dpl = RLA_SU(pin[ku + 1], pm);
            res = RLA_DV(RLA_AD(RLA_MU(oz[ku], dpl), RLA_MU(oz[ku + 1], dpu)),
                         RLA_AD(dpl, dpu));
        }
        o3[(long long)kout * oks + (long long)col * ocs] = res;
    }
}

// LW outputs for one adapter chunk of nc columns: rthratenlw =
// (hr / 86400) / pi3d on the model layers into a chunk (nz, nc) block, plus
// glw = dflx(:,0) and olr = uflx(:,ntop).  pi3d is the chunk's (nc, nz)
// column block.
extern "C" __global__ void rla_lw_out(
    int nc, int nz, int nl,
    const float* __restrict__ hr,       // (nc, nl)
    const float* __restrict__ uflx,     // (nc, nl+1)
    const float* __restrict__ dflx,     // (nc, nl+1)
    const float* __restrict__ pi3d,     // (nc, nz)
    float day_seconds,
    float* __restrict__ rthraten,       // (nz, nc)
    float* __restrict__ glw, float* __restrict__ olr)   // (nc,)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)nc * nz) return;
    int k = (int)(i / nc);
    int col = (int)(i % nc);
    float tten = RLA_DV(hr[(long long)col * nl + k], day_seconds);
    rthraten[i] = RLA_DV(tten, pi3d[(long long)col * nz + k]);
    if (k == 0) {
        glw[col] = dflx[(long long)col * (nl + 1)];
        olr[col] = uflx[(long long)col * (nl + 1) + nl];
    }
}

// SW outputs for one day chunk of nc columns: gsw = swdflx(:,0) -
// swuflx(:,0) and rthratensw = (swhr / 86400) / pi3d into a chunk (nz, nc)
// block; pi3d is the chunk's (nc, nz) column block.
extern "C" __global__ void rla_sw_out(
    int nc, int nz, int nlay,
    const float* __restrict__ swdflx,   // (nc, nlay+1)
    const float* __restrict__ swuflx,
    const float* __restrict__ swhr,     // (nc, nlay)
    const float* __restrict__ pi3d,     // (nc, nz)
    float day_seconds,
    float* __restrict__ rthraten,       // (nz, nc)
    float* __restrict__ gsw)            // (nc,)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)nc * nz) return;
    int k = (int)(i / nc);
    int j = (int)(i % nc);
    float tten = RLA_DV(swhr[(long long)j * nlay + k], day_seconds);
    rthraten[i] = RLA_DV(tten, pi3d[(long long)j * nz + k]);
    if (k == 0) {
        gsw[j] = RLA_SU(swdflx[(long long)j * (nlay + 1)],
                        swuflx[(long long)j * (nlay + 1)]);
    }
}

// Global output twins: only the input/output addresses differ.
extern "C" __global__ void rla_lw_out_grid(
    int nc, int nz, int nl,
    const float* __restrict__ hr,       // (nc, nl)
    const float* __restrict__ uflx,     // (nc, nl+1)
    const float* __restrict__ dflx,     // (nc, nl+1)
    const float* __restrict__ pi3d,     // (nz, ncol)
    float day_seconds, long long ncol, long long c0,
    float* __restrict__ rthraten,       // (nz, ncol)
    float* __restrict__ glw, float* __restrict__ olr)   // (ncol,)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)nc * nz) return;
    int k = (int)(i / nc);
    int col = (int)(i % nc);
    long long dest = (long long)k * ncol + c0 + col;
    float tten = RLA_DV(hr[(long long)col * nl + k], day_seconds);
    rthraten[dest] = RLA_DV(tten, pi3d[dest]);
    if (k == 0) {
        glw[c0 + col] = dflx[(long long)col * (nl + 1)];
        olr[c0 + col] = uflx[(long long)col * (nl + 1) + nl];
    }
}

extern "C" __global__ void rla_sw_out_grid(
    int nc, int nz, int nlay,
    const float* __restrict__ swdflx,   // (nc, nlay+1)
    const float* __restrict__ swuflx,
    const float* __restrict__ swhr,     // (nc, nlay)
    const float* __restrict__ pi3d,     // (nz, ncol)
    float day_seconds, long long ncol,
    const long long* __restrict__ idx,
    float* __restrict__ rthraten,       // (nz, ncol)
    float* __restrict__ gsw)            // (ncol,)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)nc * nz) return;
    int k = (int)(i / nc);
    int j = (int)(i % nc);
    long long col = idx[j];
    long long dest = (long long)k * ncol + col;
    float tten = RLA_DV(swhr[(long long)j * nlay + k], day_seconds);
    rthraten[dest] = RLA_DV(tten, pi3d[dest]);
    if (k == 0) {
        gsw[col] = RLA_SU(swdflx[(long long)j * (nlay + 1)],
                        swuflx[(long long)j * (nlay + 1)]);
    }
}

// WRF v4.6.1 module_radiation_driver.F:2870.  The driver first adds SW
// to the LW tendency, then recovers the diagnostic SW by subtraction.
// Keeping the raw SW scheme output loses this float32 rounding step.
extern "C" __global__ void rla_driver_sw_tendency(
    long long n, const float* __restrict__ lw, float* __restrict__ sw)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) sw[i] = RLA_SU(RLA_AD(lw[i], sw[i]), lw[i]);
}

// Driver SWDOWN: the subtraction and division round separately.
extern "C" __global__ void rla_swdown(
    long long n, const float* __restrict__ gsw,
    const float* __restrict__ albedo, float* __restrict__ swdown)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float result = RLA_DV(gsw[i], RLA_SU(1.0f, albedo[i]));
    // NumPy preserves the first NaN operand's sign/payload and quiets it.
    // Invalid division without a NaN operand returns negative quiet NaN.
    // CUDA canonicalizes NaNs, so restore these host result bits.
    unsigned int gb = __float_as_uint(gsw[i]);
    unsigned int ab = __float_as_uint(albedo[i]);
    if ((__float_as_uint(result) & 0x7fffffffU) > 0x7f800000U) {
        unsigned int bits = ((gb & 0x7fffffffU) > 0x7f800000U)
            ? (gb | 0x00400000U)
            : (((ab & 0x7fffffffU) > 0x7f800000U)
               ? (ab | 0x00400000U) : 0xffc00000U);
        result = __uint_as_float(bits);
    }
    swdown[i] = result;
}

// Radius conversion for up to three column blocks in one launch.
// Each value keeps rla_scale's operands and single rounding point.
extern "C" __global__ void rla_scale_radii(
    long long n, int count, const float* x0, const float* x1,
    const float* x2, float scale, float* out)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * count) return;
    int field = (int)(i / n);
    const float* x = field == 0 ? x0 : (field == 1 ? x1 : x2);
    out[i] = RLA_MU(x[i % n], scale);
}
