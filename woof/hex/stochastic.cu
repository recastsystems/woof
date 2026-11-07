// GPU stochastic physics kernels, written from the published methods:
//   SPPT  Buizza, Miller & Palmer (1999, QJRMS 125, 2887-2908);
//         Palmer et al. (2009, ECMWF Tech. Memo. 598).
//   SKEB  Shutts (2005, QJRMS 131, 3079-3102); Berner et al. (2009, JAS 66, 603-626).
//   Pattern: real spherical harmonics with a Gaussian variance spectrum and
//         AR(1) coefficients (Weaver & Courtier 2001, QJRMS 127, 1815-1846,
//         for the Gaussian correlation model on the sphere).
//   Saturation: Bolton (1980, MWR 108, 1046-1053).
// No code from NOAA's stochastic_physics (LGPL) was read or used.
// Philox4x32-10: Random123 BSD notice in licenses/LICENSE-Random123.txt.
__device__ __forceinline__ uint4 stoch_philox(uint4 c, unsigned long long seed) {
    unsigned int k0 = (unsigned int)seed;
    unsigned int k1 = (unsigned int)(seed >> 32);
    for (int round = 0; round < 10; ++round) {
        unsigned long long p0 = 0xD2511F53ULL * c.x;
        unsigned long long p1 = 0xCD9E8D57ULL * c.z;
        c = make_uint4((unsigned int)(p1 >> 32) ^ c.y ^ k0,
                       (unsigned int)p1,
                       (unsigned int)(p0 >> 32) ^ c.w ^ k1,
                       (unsigned int)p0);
        k0 += 0x9E3779B9U;
        k1 += 0xBB67AE85U;
    }
    return c;
}


extern "C" __global__ void harmonics(const double *lat, const double *lon,
                                     float *basis, int points, int degree) {
    long long i = (long long)blockDim.x*blockIdx.x+threadIdx.x;
    if (i >= (long long)points*(degree+1)) return;
    int point=i/(degree+1), m=i%(degree+1), modes=(degree+1)*(degree+1)-1;
    // A valid basis can exceed 2^31 float words (8 GiB). Widen before
    // multiplying, so high rows never wrap into negative addresses.
    long long row=(long long)point*modes;
    double x=sin(lat[point]), s=sqrt(fmax(0.0,1-x*x));
    double diagonal=0.28209479177387814347;
    for (int j=1; j<=m; ++j) diagonal *= -sqrt((2.0*j+1)/(2.0*j))*s;
    double a=0, b=diagonal, cs=cos(m*lon[point]), sn=sin(m*lon[point]);
    for (int l=m; l<=degree; ++l) {
        double p;
        if (l==m) p=diagonal;
        else if (l==m+1) p=sqrt(2.0*m+3)*x*b;
        else p=sqrt((4.0*l*l-1)/(1.0*l*l-m*m))*x*b
                  -sqrt(((2.0*l+1)*((l-1.0)*(l-1)-m*m))/((2.0*l-3)*(1.0*l*l-m*m)))*a;
        if (l>0) {
            int center=l*l+l-1;
            if (m==0) basis[row+center]=(float)p;
            else {
                basis[row+center+m]=(float)(1.4142135623730951*p*cs);
                basis[row+center-m]=(float)(1.4142135623730951*p*sn);
            }
        }
        a=b; b=p;
    }
}
// Spherical-harmonic synthesis evaluated on the fly at arbitrary points, so
// no (points x modes) basis is ever stored: the same law on a global,
// variable-resolution or limited-area mesh. Identical normalisation and mode
// order to `harmonics` (orthonormal real harmonics; mode l*l+l-1 is m=0,
// +m the cosine and -m the sine). One thread owns one point and walks every
// (m,l). The m-diagonal and the longitude rotation are carried in double;
// the l recurrence runs in float on a power-of-two scaled value, so a tiny
// diagonal (high m near a pole) cannot underflow to zero before the
// recurrence carries it into its oscillatory range. ra/rb hold the three-term
// recurrence coefficients at each (l,m>=0) cosine slot, built once per degree.
extern "C" __global__ void harmonic_sum(const double *lat, const double *lon,
                                        const float *coeff, const float *ra,
                                        const float *rb, float *out,
                                        int points, int degree, int channels) {
    int point=blockDim.x*blockIdx.x+threadIdx.x;
    if (point>=points) return;
    double x=sin(lat[point]), s=sqrt(fmax(0.0,1-x*x));
    double c1=cos(lon[point]), s1=sin(lon[point]);
    double cm=1, sm=0, diagonal=0.28209479177387814347;
    float xf=(float)x;
    float total[3]={0,0,0};
    for (int m=0; m<=degree; ++m) {
        if (m>0) {
            diagonal*=-sqrt((2.0*m+1)/(2.0*m))*s;
            double c=cm*c1-sm*s1;
            sm=sm*c1+cm*s1; cm=c;
        }
        if (diagonal==0) break;  // every higher order is below double range
        int exponent;
        float b=(float)frexp(diagonal,&exponent), a=0;
        // 2**exponent as a float multiplier: zero (correctly) while the
        // carried value is below float range, rebuilt only on a rescale.
        float scale=ldexpf(1.0f,exponent);
        float even[3]={0,0,0}, odd[3]={0,0,0};
        for (int l=m; l<=degree; ++l) {
            float p;
            int center=l*l+l-1;
            if (l==m) p=b;
            else {
                p=ra[center+m]*xf*b-rb[center+m]*a;
                a=b; b=p;
            }
            if (fabsf(b)>0x1p60f) {
                a*=0x1p-60f; b*=0x1p-60f; p=b; exponent+=60;
                scale=ldexpf(1.0f,exponent);
            }
            if (l==0) continue;
            float value=p*scale;
            if (m==0) {
                for (int k=0; k<channels; ++k) even[k]+=value*coeff[3*center+k];
            } else {
                float root2=1.41421356237309505f*value;
                for (int k=0; k<channels; ++k) {
                    even[k]+=root2*coeff[3*(center+m)+k];
                    odd[k]+=root2*coeff[3*(center-m)+k];
                }
            }
        }
        for (int k=0; k<channels; ++k) total[k]+=(float)cm*even[k]+(float)sm*odd[k];
    }
    for (int k=0; k<channels; ++k) out[3*(long long)point+k]=total[k];
}

extern "C" __global__ void ar1(float *coeff, const float *weights, int n,
                               float phi, float innovation,
                               unsigned long long seed, unsigned long long draw) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if (i>=n) return;
    uint4 words=stoch_philox(make_uint4(i/3,(unsigned int)draw,i%3,(unsigned int)(draw>>32)),seed);
    double u=((double)words.x+0.5)*0x1p-32;
    double v=((double)words.y+0.5)*0x1p-32;
    float noise=(float)(sqrt(-2*log(u))*cos(6.2831853071795864769*v));
    coeff[i]=phi*coeff[i]+innovation*weights[i/3]*noise;
}

__device__ __forceinline__ float sppt_factor(float pattern, float taper,
                                            float clip, int logit) {
    if (isnan(pattern)) return pattern;
    // Hard clip (Palmer et al. 2009: |r| <= 1 keeps 1+r in [0,2], so a
    // tendency is never reversed) or, with `logit`, a smooth bound of the
    // same range, clip*tanh(r/clip), whose slope at r=0 is one.
    float bounded=logit ? clip*tanhf(pattern/clip)
                        : fminf(clip, fmaxf(-clip, pattern));
    return bounded*taper;
}

// Saturation mixing ratio over liquid water, Bolton (1980) eq. 10.
__device__ __forceinline__ float stoch_qsat(float temperature, float pressure) {
    float celsius=temperature-273.15f;
    float vapour=611.2f*expf(17.67f*celsius/(celsius+243.5f));
    return 0.62198f*vapour/fmaxf(pressure-vapour, 1e-3f*pressure);
}

// Palmer et al. (2009): a perturbed humidity tendency that would make the
// air supersaturated, or the humidity negative, falls back to the
// unperturbed tendencies at that point. The candidate state after one step
// uses the perturbed theta_m and qv tendencies (dry-mass coupled sources,
// rho_zz units) and MPAS's equation of state,
// p = p0*(zz*Rd*rho_zz*theta_m/p0)**(cp/cv).
__device__ __forceinline__ bool stoch_moist_safe(float r, float rho, float rtheta,
        float q, float ttend, float qtend, float zz, float dt) {
    const float rd=287.0f, rv=461.6f, cp=1003.0f, cv=716.0f, p0=1.0e5f;
    float qstar=q+dt*(1.0f+r)*qtend/rho;
    if (!(qstar>=0)) return false;
    float theta_m=(rtheta+dt*(1.0f+r)*ttend)/rho;
    float exner=powf(zz*rd*rho*theta_m/p0, rd/cv);
    float pressure=p0*powf(exner, cp/rd);
    float temperature=theta_m/(1.0f+rv/rd*qstar)*exner;
    return qstar<=stoch_qsat(temperature, pressure);
}

// Tendencies may have padding or strided level/cell layouts. Strides are in
// float words, not bytes. Spectral and mesh workspaces are contiguous.
extern "C" __global__ void force_cells(
    float *theta, float *qv, const float *rho, const float *taper,
    const float *pattern, const float *thermal, const float *dissipation,
    long long ts0, long long ts1, long long qs0, long long qs1,
    long long rs0, long long rs1, int cells, int levels,
    int sppt, int skeb, int has_qv, int logit, float clip, float thermal_scale,
    int guard, const float *rtheta_state, const float *qv_state, const float *zz,
    long long ps0, long long ps1, long long vs0, long long vs1, float dt,
    unsigned int *reverted) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if (i>=cells*levels) return;
    int k=i/cells, c=i%cells;
    long long t=k*ts0+c*ts1;
    float value=theta[t];
    if (sppt) {
        float r=sppt_factor(pattern[c],taper[i],clip,logit);
        if (guard && has_qv && r!=0
                && !stoch_moist_safe(r, rho[k*rs0+c*rs1], rtheta_state[k*ps0+c*ps1],
                                     qv_state[k*vs0+c*vs1], value, qv[k*qs0+c*qs1],
                                     zz[i], dt)) {
            r=0;
            if (reverted) atomicAdd(reverted, 1u);
        }
        value*=1.0f+r;
        if (has_qv) qv[k*qs0+c*qs1]*=1.0f+r;
    }
    if (skeb) value+=rho[k*rs0+c*rs1]*taper[i]*thermal[c]
                    *sqrtf(thermal_scale*dissipation[k]);
    theta[t]=value;
}

extern "C" __global__ void force_edges(
    float *momentum, const float *rho, const float *taper,
    const float *vertex_taper, const float *pattern, const float *psi,
    const float *dissipation, const float *norm, const int *coe, const int *voe,
    const float *orientation, const float *dv,
    long long ms0, long long ms1, long long rs0, long long rs1,
    int cells, int vertices, int edges, int levels,
    int sppt, int skeb, int logit, float clip, float kinetic_scale) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if (i>=edges*levels) return;
    int k=i/edges, e=i%edges;
    long long m=k*ms0+e*ms1;
    float value=momentum[m];
    if (sppt) {
        int a=coe[2*e], b=coe[2*e+1];
        // A limited-area boundary edge has one cell (-1 marks the other).
        if (a<0) a=b;
        if (b<0) b=a;
        float fa=sppt_factor(pattern[a],taper[k*cells+a],clip,logit);
        float fb=sppt_factor(pattern[b],taper[k*cells+b],clip,logit);
        value*=1.0f+0.5f*(fa+fb);
    }
    // orientation is zero on an edge without both cells and both dual vertices
    // (a limited-area boundary edge): no curl there, and its edge density, which
    // native leaves undefined, is never read.
    if (skeb && orientation[e]!=0 && voe[2*e]>=0 && voe[2*e+1]>=0) {
        int a=voe[2*e], b=voe[2*e+1];
        float rotational=orientation[e]*(vertex_taper[k*vertices+b]*psi[b]
                         -vertex_taper[k*vertices+a]*psi[a])/dv[e];
        float denominator=norm[0];
        if (denominator<1e-30f) denominator=1e-30f;
        rotational/=denominator;
        value+=rho[k*rs0+e*rs1]*rotational*sqrtf(kinetic_scale*dissipation[k]);
    }
    momentum[m]=value;
}

// Class-local storage, global connectivity. Halo edges evaluate exactly the
// same dual vertices and taper as their owner, even across LTS rate classes.
extern "C" __global__ void force_lts_edges(
    float *momentum, const float *rho, const float *taper,
    const float *vertex_taper, const float *pattern, const float *psi,
    const float *dissipation, const float *norm, const int *coe, const int *voe,
    const float *orientation, const float *dv, const int *edge_l2g,
    long long ms0, long long ms1, long long rs0, long long rs1,
    int cells, int vertices, int edges, int levels,
    int sppt, int skeb, int logit, float clip, float kinetic_scale) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if (i>=edges*levels) return;
    int k=i/edges, e=i%edges, g=edge_l2g[e];
    long long m=k*ms0+e*ms1;
    float value=momentum[m];
    if (sppt) {
        int a=coe[2*g], b=coe[2*g+1];
        if (a<0) a=b;
        if (b<0) b=a;
        float fa=sppt_factor(pattern[a],taper[k*cells+a],clip,logit);
        float fb=sppt_factor(pattern[b],taper[k*cells+b],clip,logit);
        value*=1.0f+0.5f*(fa+fb);
    }
    if (skeb && orientation[g]!=0 && voe[2*g]>=0 && voe[2*g+1]>=0) {
        int a=voe[2*g], b=voe[2*g+1];
        float rotational=orientation[g]*(vertex_taper[k*vertices+b]*psi[b]
                         -vertex_taper[k*vertices+a]*psi[a])/dv[g];
        float denominator=norm[0];
        if (denominator<1e-30f) denominator=1e-30f;
        value+=rho[k*rs0+e*rs1]*(rotational/denominator)
               *sqrtf(kinetic_scale*dissipation[k]);
    }
    momentum[m]=value;
}

// Edge sinks have already been integrated once by their unique owners. Each
// global cell contributes once to the area reduction, without class halos.
extern "C" __global__ void lts_sink_cells(
    const float *integral, float *sink, const int *eoc, const int *nec,
    const float *edge_weight, int cells, int edges, int max_edges, int levels,
    float duration) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if (i>=cells*levels) return;
    int k=i/cells, c=i%cells;
    float value=0, weight=0;
    for (int j=0; j<nec[c]; ++j) {
        int e=eoc[c*max_edges+j];
        if (e<0 || e>=edges) continue;
        value+=integral[k*edges+e]*edge_weight[e];
        weight+=edge_weight[e];
    }
    sink[i]=(weight>0) ? value/(weight*duration) : 0;
}
