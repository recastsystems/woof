// WRF v4.7.1 module_diag_nwp.F:393-902, unchanged source and public-domain
// notice in tools/diag_wrf471_oracle. Native 50-bin PSD hail, not HAILCAST.
// Explicit device leaf only. No stock selector is admitted by this file.
#ifdef GPUWM_WRF_EXACT
#define nh_exp32 gfk_exp
#define nh_pow32 gfk_pow
#define nh_log10 gfk_log10
#define nh_exp64 glibc_exp
#define nh_log64 glibc_log
#define nh_pow64 glibc_pow
#else
#define nh_exp32 expf
#define nh_pow32 powf
#define nh_log10 log10f
#define nh_exp64 exp
#define nh_log64 log
#define nh_pow64 pow
#endif

__device__ float nh_wgamma(float value) {
    const double cof[6] = {76.18009172947146, -86.50532032941677,
        24.01409824083091, -1.231739572450155, .1208650973866179e-2,
        -.5395239384953e-5};
    double x = value, y = x, tmp = x+5.5;
    tmp = (x+.5)*nh_log64(tmp)-tmp;
    double ser = 1.000000000190015;
    for (int j=0;j<6;++j) { y=y+1.; ser=ser+cof[j]/y; }
    float gammln = (float)(tmp+nh_log64(2.5066282746310005*ser/x));
    return nh_exp32(gammln);
}

extern "C" __global__ void nwp_hail(
        const float* qg, const float* rho, const float* qh,
        const float* ng, const float* nh, const float* volg,
        const float* qr, const float* nr, float* hail1, float* hail2,
        int nz, int n, int mp, int has_qh, int has_ng, int has_nh, int has_vol,
        int mpuse_hail, int gsfc_hail, int gsfc_2ice,
        int nssl_2moment, int nssl_hail, float nssl_cnoh, float nssl_cnohl,
        float nssl_rho_qh, float nssl_rho_qhl, float nssl_alphah, float nssl_alphahl) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if (c>=n) return;
    float density=500.f, mu=0.f;
    double intercept=0.;
    bool fixed=false, active=false, thompson=false;
    switch(mp) {
    case 2: active=true; density=917.f; intercept=4.e4; fixed=true; break;
    case 6: active=true; intercept=4.e6; fixed=true; break;
    case 16:
        active=true; intercept=4.e6; fixed=true;
        if(mpuse_hail==1){ density=700.f; intercept=4.e4; } break;
    case 97:
        if(gsfc_2ice==0 || gsfc_2ice==2){
            active=true; density=gsfc_hail==1?900.f:400.f;
            intercept=4.e4; fixed=true;
        } break;
    case 8: case 28: active=true; thompson=!has_ng; break;
    case 38: active=true; break;
    case 10: case 40: active=true; density=mpuse_hail==1?900.f:400.f; break;
    case 9: active=true; density=900.f; break;
    case 18:
        if(nssl_2moment==0){
            active=true; fixed=true;
            density=nssl_hail==1?nssl_rho_qhl:nssl_rho_qh;
            intercept=nssl_hail==1?nssl_cnohl:nssl_cnoh;
            mu=nssl_hail==1?nssl_alphahl:nssl_alphah;
        } break;
    default: break; // Native leaves all other scheme cases without PSD hail.
    }
    if(!active) return;
    const float b=3.f;
    float a=(3.1415926536f/6.f)*density;
    float ge1=b+1.f, ge2=mu+1.f, ge3=b+mu+1.f;
    float gg1=nh_wgamma(ge1), gg2=nh_wgamma(ge2), gg3=nh_wgamma(ge3);
    const float densities[9]={50.f,100.f,200.f,300.f,400.f,500.f,600.f,700.f,800.f};
    float maximum1=hail1[c], maximum2=hail2[c];
    for(int k=0;k<nz;++k){
        size_t x=(size_t)k*n+c;
        float mass=fmaxf(1.e-12f,(has_qh?qh[x]:qg[x])*rho[x]);
        if(mass<1.e-6f) continue;
        float number=has_nh?fmaxf(1.e-8f,nh[x]*rho[x]):
                     has_ng?fmaxf(1.e-8f,ng[x]*rho[x]):1.e-8f;
        if(fixed){
            double lambda=nh_pow64(intercept*(double)a*(double)gg1/(double)mass,(double)(1.f/ge1));
            number=(float)(intercept*(double)gg2*nh_pow64(lambda,(double)(-ge2)));
        } else if(thompson){
            // Native computes rain mass/number here but neither enters its PSD.
            float y=nh_log10(fmaxf(1.e-9f,mass));
            float z=3.f+(2.f/7.f)*(y+8.f);
            z=fmaxf(2.f,fminf(z,6.f));
            double n0=(double)nh_pow32(10.f,z);
            double le=nh_pow64(n0*(double)a*(double)gg1/(double)mass,(double)(1.f/ge1));
            double lg=le*(double)nh_pow32(gg3/gg2/gg1,1.f/b);
            double n0g=n0/((double)gg2*le)*nh_pow64(lg,(double)ge2);
            number=fmaxf(1.e-8f,(float)(n0g*(double)gg2*nh_pow64(lg,(double)(-ge2))));
        }
        float am=a;
        if(has_vol){
            // Only THOMPSONGH initializes idx_bg in native WRF when volume
            // is present. The explicit API refuses other volume combinations.
            float volume=fmaxf(__fdiv_rn(mass/rho[x], densities[8]),volg[x]);
            volume=fminf(__fdiv_rn(mass/rho[x], densities[0]),volume);
            int index=(int)floorf(mass/volume*.01f+.5f)+1;
            index=max(1,min(index,9));
            if(densities[index-1]<350.f) continue;
            am=(3.1415926536f/6.f)*densities[index-1];
        }
        double lambda=(double)nh_pow32(am*gg3/gg2*number/mass,1.f/b);
        double n0=(double)(number/gg2)*nh_pow64(lambda,(double)ge2);
        double sum=0., previous=0.;
        int bin;
        for(bin=49;bin>=0;--bin){
            double d=nwp_hail_d[bin];
            double f=n0*nh_pow64(d,(double)mu)*nh_exp64(-lambda*d)*nwp_hail_width[bin];
            sum=sum+f;
            if(sum>.0005) break;
            previous=sum;
        }
        double diameter;
        if(bin>=49) diameter=nwp_hail_d[49];
        else if(nwp_hail_d[bin+1]>(double)1.e-3f)
            diameter=nwp_hail_d[bin]-(sum-.0005)/(sum-previous)*(nwp_hail_d[bin]-nwp_hail_d[bin+1]);
        else diameter=(double)1.e-4f;
        float value=(float)diameter;
        if(k==0) maximum1=fmaxf(maximum1,value);
        maximum2=fmaxf(maximum2,value);
    }
    hail1[c]=maximum1; hail2[c]=maximum2;
}
