// WRF v4.6.1 dyn_em/module_initialize_heldsuarez.F, public domain.
// Analytic initialization only. RNG samples are explicit input: native
// random_seed() deliberately does not specify a reproducible seed.
// Profiles use stride nz+1, fields use packed (k,j,i), poles are v faces.
static __device__ float hs_eta(int k, int nz) {
    return 1.0f - (float)k/(float)nz;
}
static __device__ float hs_b(float eta, int k, int nz, int hybrid, float e) {
    if (hybrid<2) return eta;
    if (k==0) return 1.0f;
    if (k==nz) return 0.0f;
    if (hybrid==3) {
        float s=glibc_sinf(0.5f*3.14159f*eta);
        return eta*(s*s);
    }
    if (eta<e) return 0.0f;
    float e2=e*e, e3=e2*e, one=1.0f-e, one2=one*one;
    float b1=(2.0f*e2)*one;
    float b2=(-e)*((4.0f-3.0f*e)-e3);
    float b3=2.0f*(1.0f-e3), b4=-(1.0f-e2), b5=one2*one2;
    return (((b1+b2*eta)+b3*(eta*eta))+b4*((eta*eta)*eta))/b5;
}
extern "C" __global__ void hs_init_coord(
    int nx, int ny, int nz, int hybrid, const float* c, float* s, float* z) {
    int k=blockIdx.x*blockDim.x+threadIdx.x, n=nz+1;
    if (k>nz) return;
    float eta=hs_eta(k,nz), ep=hs_eta(k+1,nz), em=hs_eta(k-1,nz);
    float pt=c[8]*gfk_exp(-(c[9]*c[0])/(c[3]*c[7]));
    float dp=c[8]-pt, b=hs_b(eta,k,nz,hybrid,c[1]);
    z[k]=eta;
    z[10*n+k]=b;
    z[11*n+k]=(eta-b)*dp;
    float cf;
    if (k==0) cf=1.0f;
    else if (k==nz) cf=hybrid<2?1.0f:0.0f;
    else {
        float bp=hs_b(ep,k+1,nz,hybrid,c[1]);
        float bm=hs_b(em,k-1,nz,hybrid,c[1]);
        cf=(0.5f*(bp+b)-0.5f*(b+bm))/(0.5f*(ep+eta)-0.5f*(eta+em));
    }
    z[8*n+k]=cf;
    z[9*n+k]=(1.0f-cf)*dp;
    if (k<nz) {
        float dnw=ep-eta, mid=0.5f*(ep+eta);
        float bp=hs_b(ep,k+1,nz,hybrid,c[1]);
        float bh=0.5f*(bp+b), ch=(bp-b)/(ep-eta);
        z[n+k]=mid;
        z[2*n+k]=dnw;
        z[3*n+k]=1.0f/dnw;
        z[12*n+k]=ch;
        z[13*n+k]=(1.0f-ch)*dp;
        z[14*n+k]=bh;
        z[15*n+k]=(mid-bh)*dp;
        if (k>0) {
            float dm=eta-em, dn=0.5f*(dnw+dm);
            z[4*n+k]=dn;
            z[5*n+k]=1.0f/dn;
            z[6*n+k]=(0.5f*dnw)/dn;
            z[7*n+k]=(0.5f*dm)/dn;
        }
    }
    if (k==0) {
        float d0=hs_eta(1,nz)-hs_eta(0,nz);
        float d1=hs_eta(2,nz)-hs_eta(1,nz);
        float d2=hs_eta(3,nz)-hs_eta(2,nz);
        float dn1=0.5f*(d1+d0), dn2=0.5f*(d2+d1);
        float a=((2.0f*dn1+dn2)/(dn1+dn2)*d0)/dn1;
        float b=(dn1/(dn1+dn2)*d0)/dn2;
        float dt=hs_eta(nz,nz)-hs_eta(nz-1,nz);
        float dm=hs_eta(nz-1,nz)-hs_eta(nz-2,nz);
        float dn=0.5f*(dt+dm);
        s[0]=pt;
        s[1]=((360.0f/(float)nx)*c[2])/c[10];
        s[2]=((180.0f/(float)ny)*c[2])/c[10];
        s[3]=(0.5f*d1)/dn1+a;
        s[4]=((0.5f*d0)/dn1-a)-b;
        s[5]=b;
        s[6]=(0.5f*dt+dn)/dn;
        s[7]=(-0.5f*dt)/dn;
    }
}

extern "C" __global__ void hs_init_geography(
    int nx,int ny,const float* c,float* lat,float* lon,float* tx,float* ty,
    float* f,float* e,float* vx,float* vy,float* vinv) {
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=nx*(ny+1)) return;
    int j=col/nx, i=col%nx;
    float dphi=180.0f/(float)ny;
    float latitude=((float)(j+1)-0.5f)*dphi-90.0f;
    float vlat=latitude-0.5f*dphi;
    float vscale=1.0f/glibc_cosf(vlat*c[2]);
    vx[col]=(j==0 || j==ny)?0.0f:vscale;
    vinv[col]=(j==0 || j==ny)?0.0f:1.0f/vscale;
    vy[col]=1.0f;
    if(j==ny) return;
    lat[col]=latitude;
    lon[col]=((float)(i+1)-0.5f)*(360.0f/(float)nx)-180.0f;
    tx[col]=1.0f/glibc_cosf(latitude*c[2]);
    ty[col]=1.0f;
    f[col]=(2.0f*c[11])*glibc_sinf(latitude*c[2]);
    e[col]=(2.0f*c[11])*glibc_cosf(latitude*c[2]);
}

extern "C" __global__ void hs_init_columns(
    int nx,int ny,int nz,const float* c,const float* s,const float* z,
    const float* random,float* mub,float* mu,float* pb,float* thb,float* alb,
    float* theta,float* p,float* al,float* phb,float* ph,float* tsk,float* tmn) {
    int col=blockIdx.x*blockDim.x+threadIdx.x, nc=nx*ny, n=nz+1;
    if(col>=nc) return;
    float ps=c[8]*gfk_exp(-(c[9]*0.0f/c[9])/(c[3]*c[7]));
    float mass=ps-s[0], mup=(ps-s[0])-mass;
    mub[col]=mass; mu[col]=mup;
    phb[col]=0.0f; ph[col]=0.0f;
    for(int k=0;k<nz;++k) {
        int q=k*nc+col;
        float pressure=(z[14*n+k]*(ps-s[0])+z[15*n+k])+s[0];
        pb[q]=pressure;
        float t=c[7]*gfk_pow(c[8]/pressure,c[6]);
        thb[q]=t-c[7];
        float ab=((c[3]/c[8])*(thb[q]+c[7]))*gfk_pow(pressure/c[8],c[5]);
        alb[q]=ab;
        phb[q+nc]=phb[q]-(z[2*n+k]*(z[12*n+k]*mass+z[13*n+k]))*ab;
        float pert=t*(1.0f+0.004f*(random[q]-0.5f));
        theta[q]=pert-c[7];
    }
    // Preserve even the signed zeros of the dry native moist branch.
    for(int k=nz-1;k>=0;--k) {
        int q=k*nc+col;
        float qf1=0.0f, qf2=1.0f;
        if(k==nz-1)
            p[q]=((-0.5f*((z[8*n+k+1]*mup)+qf1*(z[8*n+k+1]*mass+z[9*n+k+1])))/z[3*n+k])/qf2;
        else
            p[q]=p[q+nc]-(((z[8*n+k+1]*mup)+qf1*(z[8*n+k+1]*mass+z[9*n+k+1]))/qf2)/z[5*n+k+1];
        float alt=((c[3]/c[8])*(theta[q]+c[7]))*gfk_pow((p[q]+pb[q])/c[8],c[5]);
        al[q]=alt-alb[q];
    }
    for(int k=0;k<nz;++k) {
        int q=k*nc+col;
        float cm=z[12*n+k]*mup;
        float total=(z[12*n+k]*mass+z[13*n+k])+cm;
        ph[q+nc]=ph[q]-z[2*n+k]*(total*al[q]+cm*alb[q]);
    }
    float temp[3];
    for(int k=0;k<3;++k) {
        int q=k*nc+col;
        temp[k]=(theta[q]+c[7])*gfk_pow((p[q]+pb[q])/c[8],c[6]);
    }
    tsk[col]=(s[3]*temp[0]+s[4]*temp[1])+s[5]*temp[2];
    tmn[col]=tsk[col]-0.5f;
}
