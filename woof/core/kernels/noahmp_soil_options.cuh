// WRF-lineage v4 Noah-MP 848f54ad, module_sf_noahmpdrv.F:1712-1826.
// Public domain: licenses/LICENSE-WRF-public-domain.txt. Full SR2006 routine.
// Appended to the selected leaves image; canonical image bytes are unchanged.
namespace noahmp_soil_options {
__device__ float polynomial(const float* c,float s,float t,float o) {
    return AD(AD(AD(AD(AD(AD(MU(c[0],s),MU(c[1],t)),MU(c[2],o)),
        MU(MU(c[3],s),o)),MU(MU(c[4],t),o)),MU(MU(c[5],s),t)),c[6]);
}
__device__ float cap(float x,float lo,float hi) {
    float y=x<hi?x:hi;return lo>y?lo:y;
}
}
extern "C" __global__ void noahmp_pedotransfer_sr2006(
    const float* sand,const float* clay,const float* orgm,const float* c,
    float* out,float* updated,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=4*n)return;
    int column=i/4,k=i%4;
    float s=sand[i],t=clay[i],o=orgm[i];
    if(s<=0.0f||t<=0.0f) {s=.41f;t=.18f;}
    if(o<=0.0f)o=0.0f;
    updated[column*12+k]=s;updated[column*12+4+k]=t;updated[column*12+8+k]=o;
    float t1500=noahmp_soil_options::polynomial(c,s,t,o);
    float theta1500=AD(AD(t1500,MU(c[7],t1500)),c[8]);
    float t33=noahmp_soil_options::polynomial(c+9,s,t,o);
    float theta33=AD(AD(AD(t33,MU(MU(c[16],t33),t33)),MU(c[17],t33)),c[18]);
    float ts33=noahmp_soil_options::polynomial(c+19,s,t,o);
    float thetas33=AD(AD(ts33,MU(c[26],ts33)),c[27]);
    float et=noahmp_soil_options::polynomial(c+28,s,t,thetas33);
    float psi=AD(AD(AD(et,MU(MU(c[35],et),et)),MU(c[36],et)),c[37]);
    float smcmax=AD(AD(AD(theta33,thetas33),MU(c[38],s)),c[39]);
    float b=DV(3.816712826f,SU(r_log(theta33),r_log(theta1500)));
    float dk=MU(1930.0f,r_pow(SU(smcmax,theta33),SU(3.0f,DV(1.0f,b))));
    psi=MU(.101997f,.1f>psi?.1f:psi);dk=DV(dk,3600000.0f);
    float dw=DV(MU(MU(dk,psi),b),smcmax);
    smcmax=noahmp_soil_options::cap(smcmax,.32f,.5f);
    float ref=noahmp_soil_options::cap(theta33,.17f,smcmax);
    float wlt=noahmp_soil_options::cap(theta1500,.01f,ref);
    float* y=out+column*36+k;
    y[0]=wlt;y[4]=ref;y[8]=smcmax;
    y[12]=noahmp_soil_options::cap(b,2.5f,12.0f);
    y[16]=noahmp_soil_options::cap(psi,.03f,1.0f);
    y[20]=noahmp_soil_options::cap(dk,5.e-7f,1.e-5f);
    y[24]=noahmp_soil_options::cap(dw,1.e-6f,3.e-5f);
    y[28]=noahmp_soil_options::cap(theta1500,.01f,ref);
    y[32]=noahmp_soil_options::cap(s,.05f,.95f);
}

// Native driver :745-752, :914-918, TRANSFER_MP_PARAMETERS. A single
// resident lookup replaces independent per-property indexing launches.
extern "C" __global__ void noahmp_depth_soil_parameters(
    const float* c1,const float* c2,const float* c3,const float* c4,
    const float* ice,const int* vegetation,const long long* gj,const long long* gi,
    const bool* active,const long long* lookup,const float* table,const int* desc,
    float* out,unsigned* flags,int n,int nx,int nv,int ns,int np,int rw) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    int cell=(int)(gj[c]*nx+gi[c]);
    float raw[4]={c1[cell],c2[cell],c3[cell],c4[cell]};
    int category[4];bool bad=false,water=false;
    for(int k=0;k<4;k++) {
        bool finite=isfinite(raw[k]);bad|=!finite;
        // NINT is nearest with ties away from zero. Avoid an undefined
        // float-to-int conversion for invalid input before reporting it.
        category[k]=finite && fabsf(raw[k])<2147483000.0f ?
            (int)copysignf(floorf(AD(fabsf(raw[k]),.5f)),raw[k]):0;
        water|=category[k]==14;
    }
    if(water && ice[cell]==0.0f)for(int k=0;k<4;k++)category[k]=7;
    int veg=vegetation[cell],row[4];
    int cv=min(max(veg,1),nv);
    for(int k=0;k<4;k++) {
        int cat=category[k];int cs=min(max(cat,1),ns);
        long long ix=lookup[cv*(ns+1)+cs];
        bad|=cat<1||cat>ns||ix<0;
        row[k]=(int)(ix<0?0:ix);
    }
    bad|=category[0]==14;
    if(bad && active[c])atomicOr(flags,1u);
    for(int p=0;p<np;p++) {
        int off=desc[2*p],w=desc[2*p+1];
        for(int k=0;k<w;k++)out[off*n+c*w+k]=table[row[w==1?0:k]*rw+off+k];
    }
}
