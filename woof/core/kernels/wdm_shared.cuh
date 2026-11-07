// Shared WDM warm-rain number and stock WRF arithmetic operations.
// The arithmetic policy retains each existing scheme's coefficient words
// and math calls. WDM6 must keep its previous output bytes in both modes.
// WDM5 uses its WRF column coefficient words and WOOF's own math routines.
#ifndef GPUWM_WDM_SHARED_CUH
#define GPUWM_WDM_SHARED_CUH
struct WdmWarmNumber { float pracw, nracw, nccol, nrcol; };
__device__ __forceinline__ int wdm_checked_steps(float value) {
    float count = floorf(value);
    if (!isfinite(count) || count < 0.0f || count >= 2147483648.0f) return -1;
    return count < 1.0f ? 1 : (int)count;
}

template <typename M>
__device__ __forceinline__ WdmWarmNumber wdm_warm_collection_shared(
    float qc, float qr, float nc, float nr, float den, float dtcld,
    float lenconcr, float avedia1, float avedia2,
    float rslopec3, float rslope3r)
{
    WdmWarmNumber w = {0.0f, 0.0f, 0.0f, 0.0f};
    // pracw/nracw: accretion of cloud water by rain [LH 10, A9]
    if (qr >= lenconcr) {
        if (avedia2 >= 1.0e-4f) {
            w.nracw = fminf(3.03e3f * nc * nr * (rslopec3 + 24.0f * rslope3r),
                            nc / dtcld);
            w.pracw = fminf(M::divide(M::pi(), 6.0f) * M::divide(1000.0f, den)
                            * 3.03e3f * nc * nr * rslopec3
                            * (2.0f * rslopec3 + 24.0f * rslope3r),
                            qc / dtcld);
        } else {
            w.nracw = fminf(2.59e15f * nc * nr
                            * (2.0f * rslopec3 * rslopec3
                               + 5040.0f * rslope3r * rslope3r),
                            nc / dtcld);
            w.pracw = fminf(M::divide(M::pi(), 6.0f) * M::divide(1000.0f, den)
                            * 2.59e15f * nc * nr * rslopec3
                            * (6.0f * rslopec3 * rslopec3
                               + 5040.0f * rslope3r * rslope3r),
                            qc / dtcld);
        }
    }
    // nccol: self collection of cloud water [LH A8]
    if (avedia1 >= 1.0e-4f) {
        w.nccol = 3.03e3f * nc * nc * rslopec3;
    } else {
        w.nccol = 2.0f * 2.59e15f * nc * nc * rslopec3 * rslopec3;
    }
    // nrcol: self collection of rain drops and break-up [LH A21]
    if (qr >= lenconcr) {
        if (avedia2 < 1.0e-4f) {
            w.nrcol = 5040.0f * 2.59e15f * nr * nr * rslope3r * rslope3r;
        } else if (avedia2 >= 1.0e-4f && avedia2 < 6.0e-4f) {
            w.nrcol = 24.0f * 3.03e3f * nr * nr * rslope3r;
        } else if (avedia2 >= 6.0e-4f && avedia2 < 20.0e-4f) {
            float coecol = -2.5e3f * (avedia2 - 6.0e-4f);
            w.nrcol = 24.0f * M::exp(coecol) * 3.03e3f * nr * nr * rslope3r;
        } else {
            w.nrcol = 0.0f;
        }
    }
    return w;
}
template <typename M, typename S>
__device__ __forceinline__ float wdm_rain_evaporation_shared(
    float qr, float rh1, float work1w, float work2, float satdt, float dtcld,
    const S &rr, float *nn, float *nr)
{
    float prevp = 0.0f;
    if (qr > 0.0f) {
        float coeres = rr.r * sqrtf(rr.r * rr.rb);
        prevp = (rh1 - 1.0f) * *nr * (M::precr1() * rr.r
                                      + M::precr2() * work2 * coeres) / work1w;
        if (prevp < 0.0f) {
            prevp = fmaxf(prevp, -qr / dtcld);
            prevp = fmaxf(prevp, M::divide(satdt, 2.0f));
            if (prevp == -qr / dtcld) {
                *nn = *nn + *nr;
                *nr = 0.0f;
            }
        } else {
            prevp = fminf(prevp, M::divide(satdt, 2.0f));
        }
    }
    return prevp;
}
__device__ __forceinline__ void wdm_small_drop_collapse_shared(
    float cbrt24, float rslope_r, float *qc, float *nc, float *qr, float *nr)
{
    float avedia2 = rslope_r * cbrt24;
    if (avedia2 <= 82.0e-6f) {
        *nc = *nc + *nr;
        *nr = 0.0f;
        *qc = *qc + *qr;
        *qr = 0.0f;
    }
}
template <typename M>
__device__ __forceinline__ void wdm_activate_shared(
    float rh1, float den, float p, float xl, float cpm, float dtcld,
    float *t, float *qv, float *qc, float *nn, float *nc)
{
    if (rh1 > 1.0f) {
        float ncact = fmaxf(0.0f, (*nn + *nc)
                            * fminf(1.0f, M::pow(M::activation_divide(rh1, 1.0048f), 0.6f))
                            - *nc) / dtcld;
        ncact = fminf(ncact, fmaxf(*nn, 0.0f) / dtcld);
        float pcact = fminf(4.0f * M::pi() * 1000.0f * M::actr3() * ncact
                            / (3.0f * den), fmaxf(*qv, 0.0f) / dtcld);
        *qv = fmaxf(*qv - pcact * dtcld, 0.0f);
        *qc = fmaxf(*qc + pcact * dtcld, 0.0f);
        *nn = fmaxf(*nn - ncact * dtcld, 0.0f);
        *nc = fmaxf(*nc + ncact * dtcld, 0.0f);
        *t = *t + pcact * xl / cpm * dtcld;
    }
}

__device__ __forceinline__ void wdm_condense_shared(
    float qs1, float xl, float cpm, float dtcld,
    float *t, float *qv, float *qc, float *nn, float *nc)
{
    float work1 = (fmaxf(*qv, 1.0e-15f) - qs1)
                / (1.0f + xl * xl / (461.6f * cpm) * qs1 / (*t * *t));
    float pcond = fminf(fmaxf(work1 / dtcld, 0.0f), fmaxf(*qv, 0.0f) / dtcld);
    if (*qc > 0.0f && work1 < 0.0f)
        pcond = fmaxf(work1, -*qc) / dtcld;
    if (pcond == -*qc / dtcld) {
        *nn = *nn + *nc;
        *nc = 0.0f;
    }
    *qv = *qv - pcond * dtcld;
    *qc = fmaxf(*qc + pcond * dtcld, 0.0f);
    *t = *t + pcond * xl / cpm * dtcld;
}
template <typename M>
__device__ __forceinline__ void wdm_number_limits_shared(
    float den, float *qc, float *qi, float *qr, float *nc, float *nr)
{
    if (*qc <= 1.0e-15f) *qc = 0.0f;
    if (*qi <= 1.0e-15f) *qi = 0.0f;
    if (*qr >= 1.0e-9f && *nr >= 1.0e-2f) {
        float lamdr = M::explog((M::pidnr() * *nr) / (den * *qr), 0.33333333f);
        if (lamdr <= 2.0e3f) {
            lamdr = 2.0e3f;
            *nr = M::number(den, *qr, lamdr, M::pidnr());
        } else if (lamdr >= 5.0e4f) {
            lamdr = 5.0e4f;
            *nr = M::number(den, *qr, lamdr, M::pidnr());
        }
    }
    if (*qc >= 1.0e-15f && *nc >= 1.0e1f) {
        float lamdc = M::explog((M::pidnc() * *nc) / (den * *qc), 0.33333333f);
        if (lamdc <= 2.0e4f) {
            lamdc = 2.0e4f;
            *nc = M::number(den, *qc, lamdc, M::pidnc());
        } else if (lamdc >= 5.0e5f) {
            lamdc = 5.0e5f;
            *nc = M::number(den, *qc, lamdc, M::pidnc());
        }
    }
}
#endif
