// WRF 4.7.1 module_diag_nwp.F:211-380 and phy_prep in
// module_big_step_utilities_em.F:4861-4936. Public-domain pin recorded in
// tools/diag_wrf471_oracle/sources.json. One thread owns one column.
//
// Two entry families share every arithmetic line below. The replay ABI
// (nwp_windows, nwp_prepare) takes and returns phy_prep's full interface
// pressure, as the native module does, so the oracle compares native words.
// The resident driver (nwp_prepare_resident, nwp_window_resident) keeps
// only what the end-of-step windows read from that pressure: one bit per
// interface level, "p8w > 400 hPa", plus, under diag_print without
// microphysics, the surface interface plane that diag_print reads.  The bit is the same comparison of the same value, so
// the windows are word-identical, and a 550x550x49 domain holds 2.4 MB of
// bits instead of 60 MB of pressure (seal4: the port's 3D work did not fit
// the 16 GB headline card).  The base geopotential is read where the state
// holds it (3D or the 1D column) rather than copied, and absent qv/qg read
// as the zero they were copied from.
__device__ __forceinline__ float nwp_phb_at(const float* phb, int phb3,
                                            int k, int n, int c) {
    return phb[phb3 ? (size_t)k*n+c : (size_t)k];
}

__device__ void nwp_window_column(
        const float* w, const float* p8w, const unsigned int* below,
        const float* znw, const float* qg, int has_qg, const float* rho,
        const float* ph, const float* phb, int phb3,
        const float* u10, const float* v10, const float* refl, float* const* out,
        int nz, int ny, int nx, float dt, float xtime, int history_minutes,
        int step, int adaptive, int max_time_step) {
    int c = blockIdx.x*blockDim.x+threadIdx.x;
    int n = ny*nx;
    if (c >= n) return;
    int idump = (history_minutes*60.0f)/dt;
    float from_output = fmodf(xtime,(float)history_minutes);
    bool reset = adaptive ? (from_output > 0.0f &&
        from_output <= (max_time_step*1.05f)/60.0f) : ((step-1)%idump == 0);
    if (reset) for (int field=0;field<9;++field) out[field][c] = 0.0f;
    float mean = 0.0f, count = 0.0f, graupel = 0.0f;
    float up = out[2][c], down = out[3][c];
    for (int k=0;k<=nz;++k) {
        size_t x = (size_t)k*n+c;
        bool deep = below ? ((below[(size_t)(k>>5)*n+c] >> (k&31)) & 1u) != 0u
                          : p8w[x] > 40000.0f;
        if (deep && w[x] > up) up = w[x];
        if (deep && w[x] < down) down = w[x];
        if (znw[k] >= 0.5f && znw[k] <= 0.8f) {
            mean = mean+w[x];
            count = count+1.0f;
        }
    }
    out[2][c] = up;
    out[3][c] = down;
    for (int k=0;k<nz;++k) {
        size_t x = (size_t)k*n+c;
        float depth = (ph[x+n]+nwp_phb_at(phb,phb3,k+1,n,c))/G
                     -(ph[x]+nwp_phb_at(phb,phb3,k,n,c))/G;
        float q = has_qg ? qg[x] : 0.0f;
        graupel = graupel+q*depth*rho[x];
    }
    float wind = sqrtf(u10[c]*u10[c]+v10[c]*v10[c]);
    if (wind > out[0][c]) out[0][c] = wind;
    float accumulated_mean = out[4][c]+mean/count;
    if (step%idump == 0) accumulated_mean = accumulated_mean/idump;
    out[4][c] = accumulated_mean;
    if (graupel > out[5][c]) out[5][c] = graupel;
    // WRF samples the lowest mass level here, not the column maximum.
    if (refl[c] > out[6][c]) out[6][c] = refl[c];
    // The native module's three column outputs have no reader in the
    // resident driver, which passes no planes for them.
    if (out[9]) out[9][c] = mean;
    if (out[10]) out[10][c] = count;
    if (out[11]) out[11][c] = graupel;
}

extern "C" __global__ void nwp_windows(
        const float* w, const float* p8w, const float* znw,
        const float* qg, const float* rho, const float* ph, const float* phb,
        const float* u10, const float* v10, const float* refl, float* storage,
        int nz, int ny, int nx, float dt, float xtime, int history_minutes,
        int step, int adaptive, int max_time_step) {
    int n=ny*nx; float* out[12];
    for(int f=0;f<12;++f) out[f]=storage+(size_t)f*n;
    nwp_window_column(w,p8w,(const unsigned int*)0,znw,qg,1,rho,ph,phb,1,u10,v10,refl,out,
                      nz,ny,nx,dt,xtime,history_minutes,step,adaptive,max_time_step);
}

extern "C" __global__ void nwp_window_resident(
        const float* w, const unsigned int* below, const float* znw,
        const float* qg, int has_qg, const float* rho, const float* ph,
        const float* phb, int phb3,
        const float* u10, const float* v10, const float* refl,
        float* o0, float* o1, float* o2, float* o3, float* o4, float* o5, float* o6, float* o7, float* o8,
        int nz, int ny, int nx, float dt, float xtime, int history_minutes,
        int step, int adaptive, int max_time_step) {
    float* out[12]={o0,o1,o2,o3,o4,o5,o6,o7,o8,(float*)0,(float*)0,(float*)0};
    nwp_window_column(w,(const float*)0,below,znw,qg,has_qg,rho,ph,phb,phb3,u10,v10,refl,out,
                      nz,ny,nx,dt,xtime,history_minutes,step,adaptive,max_time_step);
}

#ifdef GPUWM_WRF_EXACT
#define nwp_log gfk_log
#define nwp_exp gfk_exp
#else
#define nwp_log logf
#define nwp_exp expf
#endif
__device__ __forceinline__ float nwp_rho_at(const float* alt, const float* qv,
                                            int has_qv, size_t x) {
    return (1.f/alt[x])*(1.f+(has_qv ? qv[x] : 0.f));
}

__device__ __forceinline__ float nwp_p8w_interior(const float* p, const float* fnm,
                                                  const float* fnp, int k, int n, int c) {
    size_t x=(size_t)k*n+c;
    return fnm[k]*p[x]+fnp[k]*p[x-n];
}

// Native full/half-level heights use two separately rounded operations.
__device__ __forceinline__ float nwp_p8w_bottom(const float* p, const float* ph,
                                                const float* phb, int phb3, int n, int c) {
    float z0=(ph[c]+nwp_phb_at(phb,phb3,0,n,c))/G;
    float z1=.5f*(z0+(ph[n+c]+nwp_phb_at(phb,phb3,1,n,c))/G);
    float z2=.5f*((ph[n+c]+nwp_phb_at(phb,phb3,1,n,c))/G+(ph[2*n+c]+nwp_phb_at(phb,phb3,2,n,c))/G);
    float w1=(z0-z2)/(z1-z2),w2=1.f-w1;
    return w1*p[c]+w2*p[n+c];
}

__device__ __forceinline__ float nwp_p8w_top(const float* p, const float* ph,
                                             const float* phb, int phb3, int nz, int n, int c) {
    float z0=(ph[(size_t)nz*n+c]+nwp_phb_at(phb,phb3,nz,n,c))/G;
    float z1=.5f*(z0+(ph[(size_t)(nz-1)*n+c]+nwp_phb_at(phb,phb3,nz-1,n,c))/G);
    float z2=.5f*((ph[(size_t)(nz-1)*n+c]+nwp_phb_at(phb,phb3,nz-1,n,c))/G
              +(ph[(size_t)(nz-2)*n+c]+nwp_phb_at(phb,phb3,nz-2,n,c))/G);
    float w1=(z0-z2)/(z1-z2),w2=1.f-w1;
    return nwp_exp(w1*nwp_log(p[(size_t)(nz-1)*n+c])
                   +w2*nwp_log(p[(size_t)(nz-2)*n+c]));
}

extern "C" __global__ void nwp_prepare(
        const float* p, const float* alt, const float* qv,
        const float* ph, const float* phb, const float* fnm, const float* fnp,
        float* rho, float* pw, float* full_phb, int nz, int n, int phb3) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=n) return;
    for(int k=0;k<=nz;++k)
        full_phb[(size_t)k*n+c]=nwp_phb_at(phb,phb3,k,n,c);
    for(int k=0;k<nz;++k) {
        size_t x=(size_t)k*n+c;
        rho[x]=nwp_rho_at(alt,qv,1,x);
        if(k>0) pw[x]=nwp_p8w_interior(p,fnm,fnp,k,n,c);
    }
    pw[c]=nwp_p8w_bottom(p,ph,phb,phb3,n,c);
    pw[(size_t)nz*n+c]=nwp_p8w_top(p,ph,phb,phb3,nz,n,c);
}

// below: (nz+1+31)/32 words per column, word-major ([word][column]).
extern "C" __global__ void nwp_prepare_resident(
        const float* p, const float* alt, const float* qv, int has_qv,
        const float* ph, const float* phb, int phb3, const float* fnm, const float* fnp,
        float* rho, unsigned int* below, float* surface_pw, int has_surface, int nz, int n) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=n) return;
    for(int k=0;k<nz;++k) {
        size_t x=(size_t)k*n+c;
        rho[x]=nwp_rho_at(alt,qv,has_qv,x);
    }
    float bottom=nwp_p8w_bottom(p,ph,phb,phb3,n,c);
    if(has_surface) surface_pw[c]=bottom;
    unsigned int bits=0u;
    for(int k=0;k<=nz;++k) {
        float pwk = k==0 ? bottom : k==nz ? nwp_p8w_top(p,ph,phb,phb3,nz,n,c)
                                          : nwp_p8w_interior(p,fnm,fnp,k,n,c);
        if(pwk > 40000.0f) bits |= 1u << (k&31);
        if((k&31)==31 || k==nz) {
            below[(size_t)(k>>5)*n+c]=bits;
            bits=0u;
        }
    }
}
