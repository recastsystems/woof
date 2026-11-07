// Public WRF v4.6.1/v4.7.1 gt_aod and module_ra_aerosol 3-D diagnostics.
// Public-domain source and table pins: tools/cloud_aerosol_wrf471_oracle/sources.json.
#ifndef GPUWM_WRF_EXACT
#undef FADD
#undef FSUB
#undef FMUL
#undef FDIV
#define FADD(a,b) ((a)+(b))
#define FSUB(a,b) ((a)-(b))
#define FMUL(a,b) ((a)*(b))
#define FDIV(a,b) ((a)/(b))
#define gfk_log logf
#endif
static __device__ float wrf_stock_gt_aod(float p, float dz8w, float t, float qv,
                                    float nwfa, float nifa)
{
    const int rind = AER3_NRH;
    float rhoa = FDIV(p, FMUL(287.0f, t));
    int t_idx = max(1, min(aer3_nint(FSUB(10.999f, FMUL(0.0333f, t))), 4));
    float qvsat = aer3_rslf(p, t);
    float rh = fminf(99.1f, fmaxf(10.1f, FMUL(FDIV(qv, qvsat), 100.0f)));
    int i1, i2;
    if (rh < 60.0f) {
        i1 = 1;
        i2 = 2;
    } else {
        int idx;
        if (rh >= 60.0f && rh < 80.0f) {
            idx = aer3_nint(FADD(FMUL(0.1f, rh), -4.0f));
        } else {
            idx = min(rind, aer3_nint(FADD(FMUL(0.2f, rh), -12.0f)));
        }
        float rh_d = FSUB(rh, AER3_GT_RH_ARR[idx - 1]);
        if (rh_d < 0.0f) {
            i1 = idx - 1;
            i2 = idx;
        } else {
            i1 = idx;
            i2 = idx + 1;
            if (i2 > rind) {
                i2 = rind;
                i1 = rind - 1;
            }
        }
    }
    float a1 = AER3_GT_RH_ARR[i1 - 1];
    float a2 = AER3_GT_RH_ARR[i2 - 1];
    float rh_f = fmaxf(0.0f, fminf(1.0f, FDIV(
        FSUB(FDIV(rh, FSUB(100.0f, rh)), FDIV(a1, FSUB(100.0f, a1))),
        FSUB(FDIV(a2, FSUB(100.0f, a2)), FDIV(a1, FSUB(100.0f, a1))))));
    float l1w = AER3_GT_LOOKUP[i1 - 1][t_idx - 1][0];
    float l2w = AER3_GT_LOOKUP[i2 - 1][t_idx - 1][0];
    float l1i = AER3_GT_LOOKUP[i1 - 1][t_idx - 1][1];
    float l2i = AER3_GT_LOOKUP[i2 - 1][t_idx - 1][1];
    float unit_bext1 = FADD(l1w, FMUL(FSUB(l2w, l1w), rh_f));
    float unit_bext3 = FADD(l1i, FMUL(FSUB(l2i, l1i), rh_f));
    float ntemp = fmaxf(1.0f, fminf(99999.E6f, nwfa));
    float aod_wfa = FMUL(FMUL(FMUL(unit_bext1, ntemp), dz8w), rhoa);
    ntemp = fmaxf(0.01f, fminf(9999.E6f, nifa));
    float aod_ifa = FMUL(FMUL(FMUL(unit_bext3, ntemp), dz8w), rhoa);
    float aod = FADD(aod_wfa, aod_ifa);
    return fmaxf(1.e-3f, aod);
}

// WRF module_ra_aerosol.F:565-575, 674-682, 752-758.
// p has nz+1 entries per column, including the caller's pressure halo.
extern "C" __global__ void wrf_stock_aer3_diagnostics(int nc, int nz,
    const float *p, const float *tau, const float *asy, const float *ssa,
    const float *aod, float *out) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    const float lower[4] = {0.7782f,0.6250f,0.4415f,0.3448f};
    const float upper[4] = {1.2420f,0.7782f,0.6250f,0.4415f};
    float ang = 0.0f, total = 0.0f, s = 0.0f, g = 0.0f;
    for (int k=0; k<nz; ++k) {
        float xy=0.0f, xx=0.0f;
        for (int b=7; b<=10; ++b) {
            float x=gfk_log(FDIV(FMUL(0.5f,FADD(lower[b-7],upper[b-7])),0.55f));
            float y=gfk_log(FDIV(tau[((size_t)c*14+b)*nz+k],aod[(size_t)c*nz+k]));
            xy=FADD(xy,FMUL(x,y)); xx=FADD(xx,FMUL(x,x));
        }
        size_t ck=(size_t)c*(nz+1)+k;
        float term=FDIV(FDIV(xy,4.0f),FDIV(xx,4.0f));
        term=FMUL(term,FSUB(p[ck],p[ck+1]));
        term=FDIV(term,FSUB(p[(size_t)c*(nz+1)],p[(size_t)c*(nz+1)+nz-1]));
        ang=FSUB(ang,term);
        total=FADD(total,aod[(size_t)c*nz+k]);
    }
    for (int b=7; b<=10; ++b) {
        s=FADD(s,ssa[((size_t)c*14+b)*nz]);
        g=FADD(g,asy[((size_t)c*14+b)*nz]);
    }
    out[c]=FDIV(s,4.0f); out[nc+c]=FDIV(g,4.0f);
    out[2*nc+c]=total; out[3*nc+c]=ang;
}

// Fast production diagnostic: one warp per column exposes level parallelism.
// Strict replay retains the native left-to-right sums above.
extern "C" __global__ void wrf_stock_aer3_diagnostics_fast(int nc, int nz, int has_halo,
    const float *p, const float *halo, const float *tau, const float *asy,
    const float *ssa, const float *aod, float *out) {
    int c=(blockIdx.x*blockDim.x+threadIdx.x)/32, lane=threadIdx.x%32;
    if(c>=nc) return;
    const float lower[4]={.7782f,.6250f,.4415f,.3448f};
    const float upper[4]={1.2420f,.7782f,.6250f,.4415f};
    float ang=0.f,total=0.f;
    for(int k=lane;k<nz;k+=32) {
        float xy=0.f,xx=0.f;
        for(int b=7;b<=10;++b) {
            float x=logf((.5f*(lower[b-7]+upper[b-7]))/.55f);
            float y=logf(tau[((size_t)c*14+b)*(nz+1)+k]/aod[(size_t)c*nz+k]);
            xy+=x*y;xx+=x*x;
        }
        float next=k+1<nz?p[(size_t)c*nz+k+1]:(has_halo?halo[c]:0.f);
        ang-=(xy/xx)*(p[(size_t)c*nz+k]-next)/(p[(size_t)c*nz]-p[(size_t)c*nz+nz-1]);
        total+=aod[(size_t)c*nz+k];
    }
    for(int d=16;d;d/=2) {
        ang+=__shfl_down_sync(0xffffffff,ang,d);
        total+=__shfl_down_sync(0xffffffff,total,d);
    }
    if(lane==0) {
        float s=0.f,g=0.f;
        for(int b=7;b<=10;++b) {s+=ssa[((size_t)c*14+b)*(nz+1)];g+=asy[((size_t)c*14+b)*(nz+1)];}
        out[c]=s*.25f;out[nc+c]=g*.25f;out[2*nc+c]=total;out[3*nc+c]=ang;
    }
}

extern "C" __global__ void wrf_stock_aer3_optics(
    int nc, int nz, int nlayers,
    const float* __restrict__ p3d, const float* __restrict__ t3d,
    const float* __restrict__ qv3d, const float* __restrict__ dz8w,
    const float* __restrict__ nwfa, const float* __restrict__ nifa,
    float* __restrict__ ztaua, float* __restrict__ zasya,
    float* __restrict__ zomga, float* __restrict__ taod)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)nc * nlayers) return;
    int c = (int)(tid / nlayers);
    int k = (int)(tid - (long long)c * nlayers);
    size_t base = (size_t)c * AER3_NB * nlayers + (size_t)k;
    if (k >= nz) {
        for (int b = 0; b < AER3_NB; ++b) {
            ztaua[base + (size_t)b * nlayers] = 0.0f;
            zasya[base + (size_t)b * nlayers] = 0.0f;
            zomga[base + (size_t)b * nlayers] = 1.0f;
        }
        return;
    }
    size_t ck = (size_t)c * nz + (size_t)k;
    float p = p3d[ck], t = t3d[ck], qv = qv3d[ck];
    float aod = wrf_stock_gt_aod(p, dz8w[ck], t, qv, nwfa[ck], nifa[ck]);
    taod[ck] = aod;
    float rh = aer3_rh(p, t, qv);
    // the Lagrange stencil: ii is the first node at or above rh
    int ii = 1;
    while (ii <= AER3_NRH && rh > AER3_RHS[ii - 1]) ++ii;
    int imin = max(1, ii - 2 - 1);
    int imax = min(AER3_NRH, ii + 2);
    float lj[6];
    for (int jj = imin; jj <= imax; ++jj) {
        float l = 1.0f;
        for (int kk = imin; kk <= imax; ++kk) {
            if (kk != jj) {
                l = FDIV(FMUL(l, FSUB(rh, AER3_RHS[kk - 1])),
                         FSUB(AER3_RHS[jj - 1], AER3_RHS[kk - 1]));
            }
        }
        lj[jj - imin] = l;
    }
    for (int b = 0; b < AER3_NB; ++b) {
        float tau = 0.0f, ssa = 0.0f, asy = 0.0f;
        for (int jj = imin; jj <= imax; ++jj) {
            float l = lj[jj - imin];
            tau = FADD(tau, FMUL(FMUL(l, AER3_RAOD[jj - 1][b]), aod));
            ssa = FADD(ssa, FMUL(l, AER3_SSA[jj - 1][b]));
            asy = FADD(asy, FMUL(l, AER3_ASY[jj - 1][b]));
        }
        ztaua[base + (size_t)b * nlayers] = tau;
        zasya[base + (size_t)b * nlayers] = asy;
        zomga[base + (size_t)b * nlayers] = ssa;
    }
}

