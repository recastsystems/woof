// Standalone WRF 4.6.1 helpers. The active mp7 entry does not call them.
// Numerical authority: module_mp_gsfcgce_4ice_nuwrf.F:1269,4508,4781.
#define G4_OPTICAL_KW 0x1.de4b5a79d9de5p-1
#define G4_OPTICAL_WRE 0x1.213d1a0000000p+3
#define G4_OPTICAL_WIM -0x1.634d6c0000000p+0
#define G4_OPTICAL_IRE 0x1.c91dae74d3b24p+0
#define G4_OPTICAL_IIM -0x1.1acb622ae704dp-13
#ifndef G4_AUX_KMAX
#define G4_AUX_KMAX 40
#endif

__device__ float g4_aux_gamma_positive(float x) {
    if (x == (float)(int)x) {
        double ga=1.;
        for(int k=2;k<=(int)x-1;++k) ga=__dmul_rn(ga,(double)k);
        return (float)ga;
    }
    return g4_gamma_toshi(x);
}

__device__ float g4_aux_radar_gamma(float xx) {
    const double cof[6]={76.18009172947146,-86.50532032941677,24.01409824083091,-1.231739572450155,.1208650973866179e-2,-.5395239384953e-5};
    double x=(double)xx,y=x,tmp=x+5.5;
    tmp=__dsub_rn(__dmul_rn(x+.5,glibc_log(tmp)),tmp);
    double ser=1.000000000190015;
    for(int j=0;j<6;++j){y=y+1.;ser=__dadd_rn(ser,__ddiv_rn(cof[j],y));}
    float ln=(float)__dadd_rn(tmp,glibc_log(__ddiv_rn(__dmul_rn(2.5066282746310005,ser),x)));
    return g4_exp(ln);
}

extern "C" __global__ void g4_aux_init(double *t,const float *constants) {
    if(blockIdx.x||threadIdx.x)return;
    const G4K &c=*(const G4K*)constants;
    t[0]=c.tnw*1.e8f;t[1]=c.tns*1.e8f;t[2]=c.tng*1.e8f;
    t[3]=3.14159f*1000.f/6.f;
    t[4]=3.14159f*100.f/6.f;
    t[5]=3.14159f*(c.roqg*1000.f)/6.f;
    t[6]=g4_aux_radar_gamma(4.f);t[7]=g4_aux_radar_gamma(7.f);
    t[8]=g4_pow(1.f/(float)t[4],1.f/3.f);
    t[9]=g4_pow(1.f/(float)t[5],1.f/3.f);
    t[10]=1.f/3.f;t[11]=1.f/3.f;
    t[12]=3.14159f*3.14159f*3.14159f*3.14159f*3.14159f;
    t[13]=0x1.a36e305532621p-14; // WRF default-real radar wavelength, folded parameter.
    // Fixed zero-temperature optical coefficients of WRF radar_init.
    // The source and the oracle's complete init-word hash pin these literals.
    t[14]=G4_OPTICAL_KW;t[15]=G4_OPTICAL_WRE;t[16]=G4_OPTICAL_WIM;
    t[17]=G4_OPTICAL_IRE;t[18]=G4_OPTICAL_IIM;
    double edges[51],weights[51];
    for(int n=0;n<51;++n)weights[n]=0.;
    for(int n=0;n<49;n+=2){weights[n]+=1./3.;weights[n+1]+=4./3.;weights[n+2]+=1./3.;}
    for(int kind=0;kind<2;++kind){
        edges[0]=100.e-6;edges[50]=(kind==0?0.02:0.05);
        for(int n=1;n<50;++n)edges[n]=glibc_exp((double)n/50.*glibc_log(edges[50]/edges[0])+glibc_log(edges[0]));
        int offset=(kind==0?19:119);
        for(int n=0;n<50;++n){t[offset+n]=sqrt(edges[n]*edges[n+1]);t[offset+50+n]=edges[n+1]-edges[n];}
    }
    for(int n=0;n<50;++n)t[219+n]=weights[n];
}

extern "C" __global__ void g4_aux_negcor(float *x,const float *rho,const float *dz,int nx,int ny,int nz,int is,int ie,int js,int je,int ks,int ke) {
    if(blockIdx.x||threadIdx.x)return;
    float a1=0.f,a2=0.f;
    for(int k=ks;k<ke;++k)for(int j=js;j<je;++j)for(int i=is;i<ie;++i){
        int idx=(k*ny+j)*nx+i;
        a1=a1+fmaxf(x[idx],0.f)*rho[idx]*dz[idx];
        a2=a2+fmaxf(-x[idx],0.f)*rho[idx]*dz[idx];
    }
    if(a1!=0.f&&a1>a2){
        float a0=(a1-a2)/a1;
        for(int k=ks;k<ke;++k)for(int j=js;j<je;++j)for(int i=is;i<ie;++i){int idx=(k*ny+j)*nx+i;x[idx]=a0*fmaxf(x[idx],0.f);}
    }
}

extern "C" __global__ void g4_aux_auto(const float *liquid,const float *number,float *rate,float *radius,int ncase) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ncase)return;
    float L=liquid[i],N=number[i];
    if(N<=0.f||L<=1.e-32f){rate[i]=0.f;return;}
    float mu=fminf(15.f,1000.f/N+2.f);
    float gfac1=g4_aux_gamma_positive(mu+4.f),gfac2=g4_aux_gamma_positive(mu+1.f);
    float gamfac=gfac1/gfac2;
    float lambda=g4_pow(4.f/3.f*3.14159f*1.f*N/L*gamfac,1.f/3.f);
    gfac1=g4_aux_gamma_positive(mu+7.f);gfac2=g4_aux_gamma_positive(mu+1.f);gamfac=gfac1/gfac2;
    float r6=g4_pow(1.f/lambda,6.f)*gamfac;
    rate[i]=1.9e11f * N * r6 * L;
    gfac1=g4_aux_gamma_positive(mu+4.f);gfac2=g4_aux_gamma_positive(mu+3.f);gamfac=gfac1/gfac2;
    radius[i]=1.f/lambda*gamfac*1.e4f;
}

__device__ double g4_aux_soak(double mass,double a,double b,double fm,cd mw,cd mi,double pi5,double lamda4) {
    fm=fmin(fmax(fm,0.),1.);double mr=.9;mr=fmin(fmax(mr,0.),1.);mr=mr+(1.-mr)*fm;
    double xw=mass*fm,d=a*glibc_pow(mass,b);if(d<1.e-12)return 0.;
    double d3=d*d*d;double vg=RPI/6.*d3;double rh=fmin(fmax(mass/vg,10.),900.);vg=mass/rh;
    double grens=1.-rh/1000.,vol;
    if(mr<=grens)vol=vg*(1.-mr*fm);
    else{double fmg=(900.-rh)/(mr*900.-rh+900.*rh/1000.);vol=fm<=fmg?(1.-mr*fm)*vg:(mass-xw)/900.+xw/1000.;}
    double large=glibc_pow(6./RPI*vol,(double)(1.f/3.f));
    double vi=(mass-xw)/(vol*900.),vw=xw/(1000.*vol),va=1.-vi-vw;
    cd m=refl_m_mix_nested(cd_make(1.,0.),mi,mw,va,vi,vw);cd m2=cd_mul(m,m);
    double k=cd_abs(cd_div(cd_sub(m2,cd_make(1.,0.)),cd_add(m2,cd_make(2.,0.))));
    double large3=large*large*large;
    return k*k*pi5*(large3*large3)/lamda4;
}

extern "C" __global__ void g4_aux_refl(const float *qv,const float *qr,const float *qs,const float *qg,const float *temp,const float *p,float *out,const double *t,int nx,int ny,int nz) {
    int col=blockIdx.x*blockDim.x+threadIdx.x,ncol=nx*ny;if(col>=ncol)return;
    float rr[G4_AUX_KMAX],rs[G4_AUX_KMAX],rg[G4_AUX_KMAX];
    double ilr[G4_AUX_KMAX],ils[G4_AUX_KMAX],ilg[G4_AUX_KMAX];
    bool lr[G4_AUX_KMAX],ls[G4_AUX_KMAX],lg[G4_AUX_KMAX];
    for(int k=0;k<nz;++k){int i=k*ncol+col;float vap=fmaxf(1.e-10f,qv[i]);float rho=.622f*p[i]/(287.f*temp[i]*(vap+.622f));
        lr[k]=qr[i]>1.e-9f;ls[k]=qs[i]>1.e-9f;lg[k]=qg[i]>1.e-9f;
        rr[k]=lr[k]?qr[i]*rho:1.e-12f;rs[k]=ls[k]?qs[i]*rho:1.e-12f;rg[k]=lg[k]?qg[i]*rho:1.e-12f;
        if(lr[k])ilr[k]=1./glibc_pow((double)((float)t[3]*(float)t[6])*t[0]/(double)rr[k],(double)(1.f/4.f));
        if(ls[k])ils[k]=1./glibc_pow((double)((float)t[4]*(float)t[6])*t[1]/(double)rs[k],(double)(1.f/4.f));
        if(lg[k])ilg[k]=1./glibc_pow((double)((float)t[5]*(float)t[6])*t[2]/(double)rg[k],(double)(1.f/4.f));
    }
    int k0=0;bool melting=false;
    for(int k=nz-2;k>=0;--k)if(temp[k*ncol+col]>273.15f&&lr[k]&&(ls[k+1]||lg[k+1])){k0=max(k+1,k0);melting=true;break;}
    const float common=(.176f/.93f)*(6.f/3.1415926536f)*(6.f/3.1415926536f);
    float sc=common*((float)t[4]/900.f)*((float)t[4]/900.f),gc=common*((float)t[5]/900.f)*((float)t[5]/900.f);
    cd mw=cd_make(t[15],t[16]),mi=cd_make(t[17],t[18]);
    for(int k=nz-1;k>=0;--k){
        float zr=1.e-22f,zs=1.e-22f,zg=1.e-22f;
        if(lr[k])zr=(float)(t[0]*t[7]*glibc_pow(ilr[k],7.));
        if(ls[k])zs=(float)((double)sc*t[1]*t[7]*glibc_pow(ils[k],7.));
        if(lg[k])zg=(float)((double)gc*t[2]*t[7]*glibc_pow(ilg[k],7.));
        if(melting&&k0>=1&&k<k0){
            if(ls[k]&&ls[k0]){double fm=fmax(.005,fmin(1.-(double)(rs[k]/rs[k0]),.99)),eta=0.,lambda=1./ils[k];
                for(int n=0;n<50;++n){double d=t[19+n],mass=t[4]*glibc_pow(d,3.);double cb=g4_aux_soak(mass,t[8],t[10],fm,mw,mi,t[12],t[13]);double fd=t[1]*glibc_pow(d,0.)*glibc_exp(-lambda*d);eta=eta+fd*cb*t[219+n]*t[69+n];}
                zs=(float)(t[13]/(t[12]*t[14])*eta);
            }
            if(lg[k]&&lg[k0]){double fm=fmax(.005,fmin(1.-(double)(rg[k]/rg[k0]),.99)),eta=0.,lambda=1./ilg[k];
                for(int n=0;n<50;++n){double d=t[119+n],mass=t[5]*glibc_pow(d,3.);double cb=g4_aux_soak(mass,t[9],t[11],fm,mw,mi,t[12],t[13]);double fd=t[2]*glibc_pow(d,0.)*glibc_exp(-lambda*d);eta=eta+fd*cb*t[219+n]*t[169+n];}
                zg=(float)(t[13]/(t[12]*t[14])*eta);
            }
        }
        out[k*ncol+col]=(float)(10.*log10((double)(zr+zs+zg)*1.e18));
    }
}
