// WRF v4.6.1 share/dfi.F: dfi_save_arrays, dfi_array_reset and
// rebalance_dfi. Public domain; pins and notice: tools/dfi_wrf461_oracle.
// RSLF is WRF's distributed phys/module_mp_thompson.F:5383-5413.
// Arrays use [species][level][column], levels bottom to top. Each column
// is independent; the hydrostatic recurrence stays inside its GPU thread.
#ifdef GPUWM_WRF_EXACT
#define DFI_POW gfk_pow
#define DFI_LOG gfk_log
#else
#define DFI_POW powf
#define DFI_LOG logf
#endif

// Snapshot descriptors: [live, saved, words, clamp_nonnegative]. Also
// used for restore by reversing the two pointers and disabling the clamp.
extern "C" __global__ void dfi_snapshot_fields(const unsigned long long* rows) {
    const unsigned long long* row = rows + 4ull * blockIdx.y;
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row[2]) return;
    const float* live = reinterpret_cast<const float*>(row[0]);
    float* saved = reinterpret_cast<float*>(row[1]);
    saved[i] = row[3] ? fmaxf(0.0f, live[i]) : live[i];
}

// Launch this for SNOW, SNOWH, SNOWC and CANWAT, before soil restoration.
extern "C" __global__ void dfi_reset_surface_water(
    const unsigned long long* rows, int save_hydrometeors) {
    const unsigned long long* row = rows + 4ull * blockIdx.y;
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row[2]) return;
    float* live = reinterpret_cast<float*>(row[0]);
    const float* saved = reinterpret_cast<const float*>(row[1]);
    live[i] = save_hydrometeors ? saved[i] : fmaxf(live[i], saved[i]);
}

// TSK, TSLB and SMOIS; RUC additionally QVG, TSNAV, SOILT1, SMFR3D and
// KEEPFR3DFLAG. SNOWC must be the result of reset_surface_water, not the
// original snow mask. Native DFI retains the evolved soil under snow.
extern "C" __global__ void dfi_reset_surface_soil(
    const unsigned long long* rows, const float* snowc,
    unsigned int columns, int save_hydrometeors) {
    const unsigned long long* row = rows + 4ull * blockIdx.y;
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row[2]) return;
    float* live = reinterpret_cast<float*>(row[0]);
    const float* saved = reinterpret_cast<const float*>(row[1]);
    if (save_hydrometeors || snowc[i % columns] == 0.0f) live[i] = saved[i];
}

__device__ __forceinline__ float dfi_rslf(float p, float t) {
    float x = fmaxf(-80.0f, t - 273.16f);
    float es = .611583699e3f + x * (.444606896e2f + x * (.143177157e1f
        + x * (.264224321e-1f + x * (.299291081e-3f + x * (.203154182e-5f
        + x * (.702620698e-8f + x * (.379534310e-11f + x * (-.321582393e-13f))))))));
    es = fminf(es, p * .15f);
    return .622f * es / (p - es);
}

__device__ __forceinline__ float dfi_temperature(
    float t, float p, float qv, int theta_m) {
    float theta = t + 300.0f;
    if (theta_m) theta = theta / (1.0f + (461.6f / 287.0f) * qv);
    return theta * DFI_POW(p / 100000.0f, 287.0f / 1004.5f);
}

extern "C" __global__ void dfi_save_relative_humidity(
    const float* t, const float* p, const float* pb, const float* qv,
    const float* qc, const float* qi, float* rh, unsigned long long words,
    int theta_m) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= words) return;
    float pressure = p[i] + pb[i];
    float qs = dfi_rslf(pressure, dfi_temperature(t[i], pressure, qv[i], theta_m));
    float out = fminf(1.0f, fmaxf(0.0f, qv[i] / qs));
    if ((qc[i] > 1.e-6f || qi[i] > 1.e-6f) && out < 1.0f) out = 1.0f;
    rh[i] = out;
}

extern "C" __global__ void dfi_restore_saturated_vapour(
    const float* t, const float* p, const float* pb, float* qv,
    const float* rh, unsigned long long words, int theta_m) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= words || rh[i] != 1.0f) return;
    float pressure = p[i] + pb[i];
    float qs = dfi_rslf(pressure, dfi_temperature(t[i], pressure, qv[i], theta_m));
    qv[i] = fmaxf(qv[i], rh[i] * qs);
}

// Pointer table (all f32): mu,mub,t,p,pb,alb,alt,al,p_hyd,ph,phb,ph0,
// psfc,moist,c1f,c2f,c1h,c2h,c3f,c4f,c3h,c4h,rdnw,rdn,dnw.
// nz is the number of physical mass levels; interfaces have nz+1 entries.
// Moisture includes only WRF's PARAM_FIRST_SCALAR:num_moist, with qv at 0.
extern "C" __global__ void dfi_rebalance_columns(
    const unsigned long long* pointers, unsigned int columns,
    unsigned int nz, unsigned int species, int theta_m,
    int hypsometric, float p_top) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= columns || nz < 2) return;
#define F(n) reinterpret_cast<float*>(pointers[n])
    const float mu = F(0)[c], mub = F(1)[c];
    float* t=F(2); float* p=F(3); const float* pb=F(4); const float* alb=F(5);
    float* alt=F(6); float* al=F(7); float* p_hyd=F(8);
    float* ph=F(9); const float* phb=F(10); float* ph0=F(11);
    float* psfc=F(12); const float* moist=F(13);
    const float *c1f=F(14), *c2f=F(15), *c1h=F(16), *c2h=F(17),
        *c3f=F(18), *c4f=F(19), *c3h=F(20), *c4h=F(21),
        *rdnw=F(22), *rdn=F(23), *dnw=F(24);
    const unsigned long long volume=(unsigned long long)nz*columns;
    ph[c]=0.0f;
    for (int k=(int)nz-1; k>=0; --k) {
        unsigned long long at=(unsigned long long)k*columns+c;
        float qtot=0.0f;
        for (unsigned int s=0; s<species; ++s) {
            unsigned long long m=s*volume+at;
            qtot=qtot+0.5f*(moist[m]+moist[m+(k==(int)nz-1 ? 0 : columns)]);
        }
        float qvf2=1.0f/(1.0f+qtot), qvf1=qtot*qvf2;
        if (k==(int)nz-1) {
            p[at]=-0.5f*(c1f[k+1]*mu+qvf1*(c1f[k+1]*mub+c2f[k+1]))/rdnw[k]/qvf2;
        } else {
            p[at]=p[at+columns]-(c1f[k+1]*mu+qvf1*(c1f[k+1]*mub+c2f[k+1]))/qvf2/rdn[k+1];
        }
        float qvf=theta_m ? 1.0f : 1.0f+(461.6f/287.0f)*moist[at];
        alt[at]=(287.0f/100000.0f)*(t[at]+300.0f)*qvf
            *DFI_POW((p[at]+pb[at])/100000.0f, -717.5f/1004.5f);
        al[at]=alt[at]-alb[at];
        p_hyd[at]=p[at]+pb[at];
    }
    if (hypsometric==1) {
        for (unsigned int k=1; k<=nz; ++k) {
            unsigned long long at=(unsigned long long)k*columns+c, below=at-columns;
            ph[at]=ph[below]-dnw[k-1]*((c1h[k-1]*mub+c2h[k-1]+c1h[k-1]*mu)*al[below]
                +(c1h[k-1]*mu)*alb[below]);
            ph0[at]=ph[at]+phb[at];
        }
    } else if (hypsometric==2) {
        ph[c]=phb[c];
        for (unsigned int k=1; k<=nz; ++k) {
            unsigned long long at=(unsigned long long)k*columns+c;
            float pfu=c3f[k]*(mub+mu)+c4f[k]+p_top;
            float pfd=c3f[k-1]*(mub+mu)+c4f[k-1]+p_top;
            float phm=c3h[k-1]*(mub+mu)+c4h[k-1]+p_top;
            ph[at]=ph[at-columns]+alt[at-columns]*phm*DFI_LOG(pfd/pfu);
        }
        for (unsigned int k=0; k<=nz; ++k) {
            unsigned long long at=(unsigned long long)k*columns+c;
            ph[at]=ph[at]-phb[at];
            ph0[at]=ph[at]+phb[at];
        }
    }
    float z0=ph0[c]/9.81f;
    float z1=0.5f*(ph0[c]+ph0[c+columns])/9.81f;
    float z2=0.5f*(ph0[c+columns]+ph0[c+2ull*columns])/9.81f;
    float w1=(z0-z2)/(z1-z2), w2=1.0f-w1;
    psfc[c]=w1*(p[c]+pb[c])+w2*(p[c+columns]+pb[c+columns]);
#undef F
}
