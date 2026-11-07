// WRF v4.7.1 module_diag_pld.F and module_diag_zld.F, pinned in
// tools/diag_wrf471_oracle/sources.json. WRF public-domain notice is retained
// beside the original sources. One CUDA thread owns a complete column.
// Arrays use ArWen (k,j,i) storage; U and V retain their native staggering.
// Output planes: U,V,T,RH,GHT,S,TD,Q[,P], each (target,j,i).
// core/level_output.py supplies the native driver clocks and held inputs.

#ifdef GPUWM_WRF_EXACT
#define ld_log gfk_log
#define ld_exp gfk_exp
#define ld_pow gfk_pow
#else
#define ld_log logf
#define ld_exp expf
#define ld_pow powf
#endif

__device__ __forceinline__ float ld_mix(float up, float down,
                                       float upper, float middle, float lower) {
    return (up * (middle - lower) + down * (upper - middle)) / (upper - lower);
}

// WRF 4.7.1 dyn_em/start_em.F:848-854 (public-domain release).
// start_em's initialization differs from phy_prep's downward integration.
extern "C" __global__ void level_initial_pressure(
        const float* p, const float* pb, const float* ht,
        const float* c3f, const float* c4f, float* pw, float* bounds,
        float p00, float t00, float a, float ptop, int nz, int n, int pb3) {
    int c = blockIdx.x*blockDim.x+threadIdx.x;
    if (c >= n) return;
    float ratio = t00/a;
    float ps = p00*ld_exp(-ratio+sqrtf(ratio*ratio-2.0f*G*ht[c]/a/RD));
    float base = pb[pb3 ? c : 0];
    pw[c] = (p[c]-base)+c3f[0]*(ps-ptop)+c4f[0]+ptop;
    float mass_hi = p[c], full_hi = pw[c];
    for (int k=1;k<=nz;++k)
    {
        pw[(size_t)k*n+c] = 2.0f*p[(size_t)(k-1)*n+c]-pw[(size_t)(k-1)*n+c];
        if (k < nz-1) mass_hi = fmaxf(mass_hi,p[(size_t)k*n+c]);
        if (k < nz) full_hi = fmaxf(full_hi,pw[(size_t)k*n+c]);
    }
    bounds[c] = mass_hi; bounds[n+c] = full_hi;
}

// The driver already holds these two phy_prep fields for PLD. Find their
// exact endpoint maxima in that same copy, avoiding another column scan.
extern "C" __global__ void level_capture_pressure(
        const float* p, const float* pw, float* held_p, float* held_pw,
        float* bounds, int nz, int n) {
    int c = blockIdx.x*blockDim.x+threadIdx.x;
    if (c >= n) return;
    float mass_hi = -__int_as_float(0x7f800000);
    float full_hi = -__int_as_float(0x7f800000);
    for (int k=0;k<nz;++k) {
        size_t x = (size_t)k*n+c;
        float pm = p[x], pf = pw[x];
        held_p[x] = pm; held_pw[x] = pf;
        if (k < nz-1) mass_hi = fmaxf(mass_hi,pm);
        full_hi = fmaxf(full_hi,pf);
    }
    held_pw[(size_t)nz*n+c] = pw[(size_t)nz*n+c];
    bounds[c] = mass_hi; bounds[n+c] = full_hi;
}

__device__ __forceinline__ float ld_dewpoint(float q, float p) {
    float e = fmaxf(q * p * 0.01f / (0.622f + q), 0.001f);
    // s3 is a REAL PARAMETER folded by Fortran, including the multiply.
    const float s3 = 0.6112f * 10.0f;
    return 273.15f + (243.5f / ((17.67f / ld_log(e / s3)) - 1.0f));
}

__device__ __forceinline__ float ld_rh(float q, float p, float t) {
    float es = 611.0f * ld_exp(5418.12f * (1.0f / 273.0f - 1.0f / t));
    float qs = 0.622f * es / (p - es);
    return q / qs * 100.0f;
}

__device__ __forceinline__ float ld_u(const float* u, int k, int j, int i,
                                     int ny, int nx) {
    size_t x = I3(k,j,i,ny,nx+1);
    return 0.5f * (u[x] + u[x+1]);
}

__device__ __forceinline__ float ld_v(const float* v, int k, int j, int i,
                                     int ny, int nx) {
    size_t x = I3(k,j,i,ny+1,nx);
    return 0.5f * (v[x] + v[x+nx]);
}

__device__ __forceinline__ float ld_speed(const float* u, const float* v,
                                         int k, int j, int i, int ny, int nx) {
    size_t x = I3(k,j,i,ny,nx+1), y = I3(k,j,i,ny+1,nx);
    float us = u[x] + u[x+1], vs = v[y] + v[y+nx];
    return 0.5f * sqrtf(us * us + vs * vs);
}

// Native interpolation of the mass-level variables. Coordinate can be p or z.
// Pressure arguments remain endpoint total/hydrostatic p as chosen by caller.
__device__ __forceinline__ void ld_mass(
        const float* u, const float* v, const float* t, const float* qv,
        float* out, int k, int j, int i, int ny, int nx, int nl, int target,
        float pu, float pd, float upper, float middle, float lower, float pm) {
    size_t n = (size_t)ny*nx, x = (size_t)k*n+j*nx+i;
    size_t plane = (size_t)nl*n, dst = (size_t)target*n+j*nx+i;
    float tu = (t[x+n]+T0)*ld_pow(pu/P0,RCP);
    float td = (t[x]+T0)*ld_pow(pd/P0,RCP);
    float tm = ld_mix(tu,td,upper,middle,lower);
    out[2*plane+dst] = tm;
    out[5*plane+dst] = ld_mix(ld_speed(u,v,k+1,j,i,ny,nx),
        ld_speed(u,v,k,j,i,ny,nx),upper,middle,lower);
    out[dst] = ld_mix(ld_u(u,k+1,j,i,ny,nx),ld_u(u,k,j,i,ny,nx),upper,middle,lower);
    out[plane+dst] = ld_mix(ld_v(v,k+1,j,i,ny,nx),ld_v(v,k,j,i,ny,nx),upper,middle,lower);
    float qu = fmaxf(qv[x+n],0.0f), qd = fmaxf(qv[x],0.0f);
    float qm = ld_mix(qu,qd,upper,middle,lower);
    out[7*plane+dst] = qm;
    out[6*plane+dst] = ld_mix(ld_dewpoint(qu,pu),ld_dewpoint(qd,pd),upper,middle,lower);
    out[3*plane+dst] = ld_rh(qm,pm,tm);
}

extern "C" __global__ void pressure_levels(
        const float* u, const float* v, const float* t, const float* qv,
        const float* ph, const float* phb, const float* pp, const float* pb,
        const float* p, const float* pw, const float* levels, float* out,
        int nz, int ny, int nx, int nl, int pressure_kind, int extrapolate,
        float missing, const float* bounds, int have_bounds) {
    int c = blockIdx.x*blockDim.x+threadIdx.x;
    int n = ny*nx;
    if (c >= n) return;
    int j = c/nx, i = c%nx, half = 0, full = 0;
    size_t plane = (size_t)nl*n;
#ifndef GPUWM_WRF_EXACT
    // A target outside all endpoint bounds cannot match any native bracket.
    // Compute bounds once rather than exhaust both searches for every
    // below-ground target. This also handles nonmonotone input columns;
    // neither endpoint monotonicity nor target ordering is assumed.
    // Only below-ground rejection earns its scan in real forecasts. Above
    // the model top, retain the native unsuccessful search rather than pay
    // for two more dependent reductions on every ordinary column.
    float mass_hi = -__int_as_float(0x7f800000);
    float full_hi = -__int_as_float(0x7f800000);
    if (have_bounds && pressure_kind == 2) {
        mass_hi = bounds[c]; full_hi = bounds[n+c];
    } else {
        for (int k=0;k<nz;++k) {
            size_t x = (size_t)k*n+c;
            if (k < nz-1) {
                mass_hi = fmaxf(mass_hi, pressure_kind == 1 ? pp[x]+pb[x] : p[x]);
            }
            full_hi = fmaxf(full_hi, pw[x]);
        }
    }
#endif
    for (int l=0;l<nl;++l) {
        size_t dst = (size_t)l*n+c;
        for (int field=0;field<8;++field) out[(size_t)field*plane+dst] = missing;
        float pm = levels[l];
        int mass_begin = half;
#ifndef GPUWM_WRF_EXACT
        if (extrapolate == 1 && pm > mass_hi) mass_begin = nz;
#endif
        for (int k=mass_begin;k<nz-1;++k) {
            size_t x = (size_t)k*n+c;
            float pu = pressure_kind == 1 ? pp[x+n]+pb[x+n] : p[x+n];
            float pd = pressure_kind == 1 ? pp[x]+pb[x] : p[x];
            if (extrapolate == 2 && k == half && pm > pd) {
                float zu = 0.5f * (ph[x]+phb[x]+ph[x+n]+phb[x+n]) / G;
                float tu = (t[x]+T0)*ld_pow(pd/P0,RCP);
                float qu = fmaxf(qv[x],0.0f);
                float tvu = tu*(1.0f+0.608f*qu);
                float gammas = 0.0f;
                if (zu > 75.0f) {
                    float tvd = tvu+zu*6.5e-3f;
                    if (tvd > 290.66f) {
                        if (tvu > 290.66f) {
                            float d = tvu-290.66f;
                            tvd = 290.66f-5.e-3f*(d*d);
                        } else tvd = 290.66f;
                    }
                    gammas = (tvu-tvd)/zu;
                }
                // r_d/g is folded as a float32 REAL PARAMETER expression.
                const float rovg = RD/G;
                float part = rovg*(ld_log(pm)-ld_log(pd));
                float z = zu-tvu*part/(1.0f+0.5f*gammas*part);
                out[4*plane+dst] = z;
                float tm = tu+(zu-z)*6.5e-3f;
                out[2*plane+dst] = tm;
                out[5*plane+dst] = ld_speed(u,v,k,j,i,ny,nx);
                out[dst] = ld_u(u,k,j,i,ny,nx);
                out[plane+dst] = ld_v(v,k,j,i,ny,nx);
                float rh = ld_rh(fmaxf(qv[x],0.0f),pd,tu);
                out[3*plane+dst] = rh;
                float es = 611.0f*ld_exp(5418.12f*(1.0f/273.0f-1.0f/tm));
                float qs = 0.622f*es/(pm-es);
                float qm = rh*qs/100.0f;
                out[7*plane+dst] = qm;
                out[6*plane+dst] = ld_dewpoint(qm,pm);
                break;
            } else if (pd >= pm && pu < pm) {
                ld_mass(u,v,t,qv,out,k,j,i,ny,nx,nl,l,pu,pd,pu,pm,pd,pm);
                half = k;
                break;
            }
        }
        int full_begin = full;
#ifndef GPUWM_WRF_EXACT
        if (pm > full_hi) full_begin = nz;
#endif
        for (int k=full_begin;k<nz;++k) {
            size_t x = (size_t)k*n+c;
            if (pw[x] >= levels[l] && pw[x+n] < levels[l]) {
                float pu = ld_log(pw[x+n]), middle = ld_log(levels[l]), pd = ld_log(pw[x]);
                float zu = (ph[x+n]+phb[x+n])/G, zd = (ph[x]+phb[x])/G;
                out[4*plane+dst] = ld_mix(zu,zd,pu,middle,pd);
                full = k;
                break;
            }
        }
    }
}

extern "C" __global__ void height_levels(
        const float* u, const float* v, const float* t, const float* qv,
        const float* ph, const float* phb, const float* pp, const float* pb,
        const float* terrain, const float* levels, float* out,
        int nz, int ny, int nx, int nl, float missing) {
    int c = blockIdx.x*blockDim.x+threadIdx.x;
    int n = ny*nx;
    if (c >= n) return;
    int j = c/nx, i = c%nx, half = 0, full = 0;
    size_t plane = (size_t)nl*n;
    for (int l=0;l<nl;++l) {
        size_t dst = (size_t)l*n+c;
        for (int field=0;field<9;++field) out[(size_t)field*plane+dst] = missing;
        // Match native '< 1', including zero and positive fractions of a metre.
        float zm = levels[l] < 1.0f ? fabsf(levels[l])+terrain[c] : levels[l];
        for (int k=half;k<nz-1;++k) {
            size_t x = (size_t)k*n+c;
            float zu = (ph[x+n]+phb[x+n]+ph[x+2*n]+phb[x+2*n])/2.0f/G;
            float zd = (ph[x]+phb[x]+ph[x+n]+phb[x+n])/2.0f/G;
            if (zd <= zm && zu > zm) {
                float pu = pp[x+n]+pb[x+n], pd = pp[x]+pb[x];
                float pm = ld_exp(ld_mix(ld_log(pu),ld_log(pd),zu,zm,zd));
                out[8*plane+dst] = pm;
                ld_mass(u,v,t,qv,out,k,j,i,ny,nx,nl,l,pu,pd,zu,zm,zd,pm);
                half = k;
                break;
            }
        }
        for (int k=full;k<nz;++k) {
            size_t x = (size_t)k*n+c;
            float zu = (ph[x+n]+phb[x+n])/G, zd = (ph[x]+phb[x])/G;
            if (zd <= zm && zu > zm) {
                // Do not replace by zm: native rounding is observable.
                out[4*plane+dst] = ld_mix(zu,zd,zu,zm,zd);
                full = k;
                break;
            }
        }
    }
}
