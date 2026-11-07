// Native WRF V3.5 phys/module_wind_fitch.F, public domain.
// Original immutable source, licence and capture recipe:
// tools/wind_legacy_wrf35_oracle. Distinct analytic physics, never table-fit.
__device__ float legacy_area(float z1,float z2,float d,float density) {
    float r=d/2.0f; z1=r-z1; z2=r-z2;
    float a=fabsf(z1),b=fabsf(z2),area;
    if(z1>0.0f && z2>0.0f)
        area=a*sqrtf(r*r-a*a)+r*r*glibc_asinf(a/r)
            -(b*sqrtf(r*r-b*b)+r*r*glibc_asinf(b/r));
    else if(z1<0.0f && z2<0.0f)
        area=b*sqrtf(r*r-b*b)+r*r*glibc_asinf(b/r)
            -(a*sqrtf(r*r-a*a)+r*r*glibc_asinf(a/r));
    else area=b*sqrtf(r*r-b*b)+r*r*glibc_asinf(b/r)
             +a*sqrtf(r*r-a*a)+r*r*glibc_asinf(a/r);
    return area*density;
}
__device__ float legacy_cube(float speed) {
#ifdef GPUWM_WIND_FAST
    return (speed*speed)*speed;
#else
    // The native real exponent speed**3. calls glibc powf under strict flags.
    return gfk_pow(speed,3.0f);
#endif
}
__device__ void legacy_coeff(float speed,const float* p,float& tk,float& th) {
    float d=p[1],standing=p[2],rated=p[3],ci=p[4],co=p[5];
    float area=3.14159265358979323846f/4.0f*(d*d);
    float mid=0.5f*(co+ci),half=0.5f*(co-mid);
#ifdef GPUWM_WIND_FAST
    float hyper=tanhf((speed-(mid-half))/(half*0.60f));
#else
    // glibc 2.43 tanhf is correctly rounded on the captured native host.
    float hyper=__double2float_rn(tanh(double((speed-(mid-half))/(half*0.60f))));
#endif
    float power=rated*(0.5f*hyper+0.5f)*0.8f,cp;
    if(speed<=ci || speed>=co) {cp=0.0f;th=standing;}
    else {
        cp=power*2.0e6f/(legacy_cube(speed)*area);
        if(speed<ci*2.0f) {
            float delta=speed-ci*2.0f;
            cp=cp*gfk_exp(-(delta*delta/(ci*2.0f)));
        }
        cp=fminf(cp,0.55f);
        th=fmaxf(fminf(cp+cp*0.75f,0.9f),standing);
    }
    tk=fmaxf(th-cp,0.0f);
}
extern "C" __global__ void legacy_locate(const float* ll,const float* latv,const float* lonu,
    int* col,int nt,int nx,int ny) {
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=nt) return;
    int found=-1;
    // Native row/column comparison, including its last-matching-cell rule.
    for(int j=0;j<ny;++j) for(int i=0;i<nx;++i)
        if(latv[j*nx+i]<=ll[2*t] && ll[2*t]<latv[(j+1)*nx+i]
           && lonu[j*(nx+1)+i]<=ll[2*t+1] && ll[2*t+1]<lonu[j*(nx+1)+i+1]) found=j*nx+i;
    col[t]=found;
}
extern "C" __global__ void legacy_drag(const int* columns,const int* ptr,const int* members,
    const float* params,const float* u,const float* v,const float* dz,const float* phb,
    float* du,float* dv,float* qke,float* qadv,float* tke_rate,float* tke_increment,
    int nc,int nz,float dx,float dt,float percell,int advect) {
    int c=columns[blockIdx.x],k=threadIdx.x;
    if(k>=nz) return;
    int n=k*nc+c;
    float before=qke[n];
    float density=percell/(dx*dx);
    for(int pos=ptr[c];pos<ptr[c+1];++pos) {
        const float* p=params+6*members[pos];
        float bottom=p[0]-p[1]/2.0f,top=p[0]+p[1]/2.0f;
        float lower=0.0f; int kb=-1,kt=-1;
        for(int z=0;z<nz;++z) {
            float upper=lower+dz[z*nc+c];
            if(bottom>=lower && bottom<=upper) kb=z;
            if(top>=lower && top<=upper) {kt=z;break;}
            lower=upper;
        }
        if(kt<0 || k<kb || k>kt) continue;
        float z1=phb[n]/9.81f-bottom-phb[c]/9.81f;
        float z2=phb[n+nc]/9.81f-bottom-phb[c]/9.81f;
        z1=fmaxf(z1,0.0f); z2=fminf(z2,p[1]);
        float speed=sqrtf(u[n]*u[n]+v[n]*v[n]),tk,th;
        legacy_coeff(speed,p,tk,th);
        float area=legacy_area(z1,z2,p[1],density);
        qke[n]=qke[n]+legacy_cube(speed)*area*tk*dt/dz[n];
        du[n]=du[n]-0.5f*u[n]*th*speed*area/dz[n];
        dv[n]=dv[n]-0.5f*v[n]*th*speed*area/dz[n];
    }
    float inc=0.5f*(qke[n]-before);
    tke_increment[n]=inc; tke_rate[n]=inc/dt;
    if(advect) qadv[n]=qke[n];
}
