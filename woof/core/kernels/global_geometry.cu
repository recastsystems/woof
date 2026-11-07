// WRF v4.6.1 module_big_step_utilities_em.F and module_small_step_em.F.
// Public domain; exact native operation ordering under strict numerics.
static __device__ __forceinline__ int gx(int i,int nx) { return (i+nx)%nx; }
static __device__ __forceinline__ int gy(int j,int ny) { return max(0,min(ny-1,j)); }
static __device__ __forceinline__ size_t gc(int k,int j,int i,int nx,int ny) {
    return ((size_t)k*ny+gy(j,ny))*nx+gx(i,nx);
}

// Dry phy_prep: the bottom pressure is WRF's linear extrapolation in z,
// not MUB+MU+PTOP. Keep both height divisions before forming the weights.
extern "C" __global__ void global_physics_prepare(
    int nx,int ny,int nz,float rcp,const float* p,const float* pb,
    const float* ph,const float* phb,const float* t,
    float* pressure,float* exner,float* temperature,float* ps) {
    int q=blockIdx.x*blockDim.x+threadIdx.x,mp=nx*ny;
    if(q<nz*mp) {
        float full=p[q]+pb[q];pressure[q]=full;
        float pi=gfk_pow(full/100000.0f,rcp);
        exner[q]=pi;temperature[q]=(t[q]+300.0f)*pi;
    }
    if(q<mp) {
        float z0=(ph[q]+phb[q])/9.81f;
        float zw1=(ph[mp+q]+phb[mp+q])/9.81f;
        float zw2=(ph[2*mp+q]+phb[2*mp+q])/9.81f;
        float z1=0.5f*(z0+zw1),z2=0.5f*(zw1+zw2);
        float w1=(z0-z2)/(z1-z2),w2=1.0f-w1;
        ps[q]=w1*(p[q]+pb[q])+w2*(p[mp+q]+pb[mp+q]);
    }
}

extern "C" __global__ void global_hold_heating(
    int nx,int ny,int nz,const float* c1,const float* c2,const float* mut,
    const float* heating,float* held) {
    int q=blockIdx.x*blockDim.x+threadIdx.x,mp=nx*ny;
    if(q<nz*mp)held[q]=(c1[q/mp]*mut[q%mp]+c2[q/mp])*heating[q];
}

extern "C" __global__ void global_add_held(
    int n,const float* ty,const float* held,float* tendency,int mp) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q<n)tendency[q]=tendency[q]+held[q]/ty[q%mp];
}
// mode 0: calc_mu_uv; mode 1: calc_mu_uv_1; mode 2: small_step_prep
// reference masses. The three associations intentionally remain distinct.
extern "C" __global__ void global_mass_faces(
    int nx,int ny,int mode,const float* mu,const float* mub,
    float* mut,float* muu,float* muv) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q<nx*ny) mut[q]=mub[q]+mu[q];
    if(q<ny*(nx+1)) {
        int j=q/(nx+1),i=q%(nx+1),a=j*nx+gx(i,nx),b=j*nx+gx(i-1,nx);
        muu[q]=mode==1?0.5f*(mu[a]+mu[b]):
            mode==2?0.5f*(mub[a]+mu[a]+mub[b]+mu[b]):
                    0.5f*(mu[a]+mu[b]+mub[a]+mub[b]);
    }
    if(q<(ny+1)*nx) {
        int j=q/nx,i=q%nx,a=gy(j,ny)*nx+i,b=gy(j-1,ny)*nx+i;
        muv[q]=mode==1?0.5f*(mu[a]+mu[b]):
            mode==2?0.5f*(mub[a]+mu[a]+mub[b]+mu[b]):
                    0.5f*(mu[a]+mu[b]+mub[a]+mub[b]);
    }
}

extern "C" __global__ void global_couple_momentum(
    int nx,int ny,int nz,const float* c1h,const float* c2h,
    const float* c1f,const float* c2f,const float* uy,const float* vxinv,
    const float* ty,const float* muu,const float* muv,const float* mut,
    const float* u,const float* v,const float* w,float* ru,float* rv,float* rw) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    int up=ny*(nx+1),vp=(ny+1)*nx,mp=nx*ny;
    if(q<nz*up) {int k=q/up,xy=q%up;ru[q]=u[q]*(c1h[k]*muu[xy]+c2h[k])/uy[xy];}
    if(q<nz*vp) {int k=q/vp,xy=q%vp;rv[q]=v[q]*(c1h[k]*muv[xy]+c2h[k])*vxinv[xy];}
    if(q<(nz+1)*mp) {int k=q/mp,xy=q%mp;rw[q]=w[q]*(c1f[k]*mut[xy]+c2f[k])/ty[xy];}
}

extern "C" __global__ void global_omega(
    int nx,int ny,int nz,float rdx,float rdy,const float* c1,const float* c2,
    const float* dnw,const float* tx,const float* uy,const float* vxinv,
    const float* mu,const float* mub,const float* u,const float* v,float* divv,float* ww) {
    int xy=blockIdx.x*blockDim.x+threadIdx.x,mp=nx*ny;
    if(xy>=mp) return;
    int j=xy/nx,i=xy%nx,ip=j*nx+gx(i+1,nx),im=j*nx+gx(i-1,nx);
    int jp=gy(j+1,ny)*nx+i,jm=gy(j-1,ny)*nx+i;
    // calc_ww_cp's private masses use a different association from calc_mu_uv.
    float mu0=0.5f*(mu[xy]+mub[xy]+mu[im]+mub[im]);
    float mu1=0.5f*(mu[ip]+mub[ip]+mu[xy]+mub[xy]);
    float mv0=0.5f*(mu[xy]+mub[xy]+mu[jm]+mub[jm]);
    float mv1=0.5f*(mu[jp]+mub[jp]+mu[xy]+mub[xy]);
    int up=ny*(nx+1),vp=(ny+1)*nx,u0=j*(nx+1)+i,v0=j*nx+i;
    float dmdt=0.0f;
    for(int k=0;k<nz;++k) {
        float value=tx[xy]*dnw[k]*(
            rdx*((c1[k]*mu1+c2[k])*u[k*up+u0+1]/uy[u0+1]
                -(c1[k]*mu0+c2[k])*u[k*up+u0]/uy[u0])
            +rdy*((c1[k]*mv1+c2[k])*v[k*vp+v0+nx]*vxinv[v0+nx]
                 -(c1[k]*mv0+c2[k])*v[k*vp+v0]*vxinv[v0]));
        divv[k*mp+xy]=value;dmdt=dmdt+value;
    }
    ww[xy]=0.0f;ww[nz*mp+xy]=0.0f;
    for(int k=1;k<nz;++k)
        ww[k*mp+xy]=ww[(k-1)*mp+xy]-dnw[k-1]*c1[k-1]*dmdt-divv[(k-1)*mp+xy];
}

extern "C" __global__ void global_small_prep_mass(
    int n,const float* mub,const float* mu0,float* mu,float* mu_save,float* muts) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=n)return;
    mu_save[q]=mu[q];mu[q]=mu0[q]-mu[q];muts[q]=mub[q]+mu0[q];
}

extern "C" __global__ void global_small_prep(
    int nx,int ny,int nz,float cpovcv,
    const float* c1h,const float* c2h,const float* c1f,const float* c2f,
    const float* uy,const float* vxinv,const float* ty,
    const float* muu,const float* muv,const float* mut,
    const float* muus,const float* muvs,const float* muts,
    const float* u0,const float* v0,const float* w0,const float* t0,const float* ph0,
    const float* pb,const float* p,const float* alt,
    float* u,float* v,float* w,float* t,float* ph,
    float* us,float* vs,float* ws,float* ts,float* phs,float* c2a) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    int up=ny*(nx+1),vp=(ny+1)*nx,mp=ny*nx;
    if(q<nz*up) {int k=q/up,xy=q%up;us[q]=u[q];
        u[q]=((c1h[k]*muus[xy]+c2h[k])*u0[q]-(c1h[k]*muu[xy]+c2h[k])*u[q])/uy[xy];}
    if(q<nz*vp) {int k=q/vp,xy=q%vp;vs[q]=v[q];
        v[q]=((c1h[k]*muvs[xy]+c2h[k])*v0[q]-(c1h[k]*muv[xy]+c2h[k])*v[q])*vxinv[xy];}
    if(q<nz*mp) {int k=q/mp,xy=q%mp;ts[q]=t[q];
        t[q]=(c1h[k]*muts[xy]+c2h[k])*t0[q]-(c1h[k]*mut[xy]+c2h[k])*t[q];
        c2a[q]=cpovcv*(pb[q]+p[q])/alt[q];}
    if(q<(nz+1)*mp) {int k=q/mp,xy=q%mp;ws[q]=w[q];
        w[q]=((c1f[k]*muts[xy]+c2f[k])*w0[q]-(c1f[k]*mut[xy]+c2f[k])*w[q])/ty[xy];
        phs[q]=ph[q];ph[q]=ph0[q]-ph[q];}
}

extern "C" __global__ void global_small_finish(
    int nx,int ny,int nz,
    const float* c1h,const float* c2h,const float* c1f,const float* c2f,
    const float* uy,const float* vx,const float* ty,
    const float* muu,const float* muv,const float* mut,
    const float* muus,const float* muvs,const float* muts,
    const float* us,const float* vs,const float* ws,const float* ts,
    const float* phs,const float* mus,const float* ww1,
    float* u,float* v,float* w,float* t,float* ph,float* mu,float* ww) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    int up=ny*(nx+1),vp=(ny+1)*nx,mp=ny*nx;
    if(q<nz*up) {int k=q/up,xy=q%up;
        u[q]=(uy[xy]*u[q]+us[q]*(c1h[k]*muu[xy]+c2h[k]))/(c1h[k]*muus[xy]+c2h[k]);}
    if(q<nz*vp) {int k=q/vp,xy=q%vp;
        v[q]=(vx[xy]*v[q]+vs[q]*(c1h[k]*muv[xy]+c2h[k]))/(c1h[k]*muvs[xy]+c2h[k]);}
    if(q<nz*mp) {int k=q/mp,xy=q%mp;
        t[q]=(t[q]+ts[q]*(c1h[k]*mut[xy]+c2h[k]))/(c1h[k]*muts[xy]+c2h[k]);}
    if(q<(nz+1)*mp) {int k=q/mp,xy=q%mp;
        w[q]=(ty[xy]*w[q]+ws[q]*(c1f[k]*mut[xy]+c2f[k]))/(c1f[k]*muts[xy]+c2f[k]);
        ph[q]=ph[q]+phs[q];ww[q]=ww[q]+ww1[q];}
    if(q<mp) mu[q]=mu[q]+mus[q];
}

extern "C" __global__ void global_slow_pressure(
    int nx,int ny,int nz,float rdx,float rdy,
    const float* c1,const float* c2,const float* ux,const float* uy,const float* vx,const float* vy,
    const float* muu,const float* muv,
    const float* ph,const float* alt,const float* p,const float* pb,const float* al,
    float* ru_tend,float* rv_tend) {
    int q=blockIdx.x*blockDim.x+threadIdx.x,mp=nx*ny,up=ny*(nx+1),vp=(ny+1)*nx;
    if(q<nz*up) {
        int k=q/up,j=(q/(nx+1))%ny,i=q%(nx+1),xy=j*(nx+1)+i;
        size_t a=gc(k,j,i,nx,ny),b=gc(k,j,i-1,nx,ny);
        float dp=(ux[xy]/uy[xy])*0.5f*rdx*(c1[k]*muu[xy]+c2[k])*
            ((ph[a+mp]-ph[b+mp]+ph[a]-ph[b])
             +(alt[a]+alt[b])*(p[a]-p[b])+(al[a]+al[b])*(pb[a]-pb[b]));
        ru_tend[q]=ru_tend[q]-dp;
    }
    if(q<nz*vp) {
        int k=q/vp,j=(q/nx)%(ny+1),i=q%nx,xy=j*nx+i;
        if(j==0 || j==ny)return;
        size_t a=gc(k,j,i,nx,ny),b=gc(k,j-1,i,nx,ny);
        float dp=(vy[xy]/vx[xy])*0.5f*rdy*(c1[k]*muv[xy]+c2[k])*
            ((ph[a+mp]-ph[b+mp]+ph[a]-ph[b])
             +(alt[a]+alt[b])*(p[a]-p[b])+(al[a]+al[b])*(pb[a]-pb[b]));
        rv_tend[q]=rv_tend[q]-dp;
    }
}

static __device__ __forceinline__ float gd(
    const float* ph,int k,int j,int i,int axis,int nx,int ny,int order) {
    int dx=axis==0,dy=axis==1;
    float d1=ph[gc(k,j+dy,i+dx,nx,ny)]-ph[gc(k,j-dy,i-dx,nx,ny)];
    if(order<=2)return d1;
    float d2=ph[gc(k,j+2*dy,i+2*dx,nx,ny)]-ph[gc(k,j-2*dy,i-2*dx,nx,ny)];
    if(order<=4)return 8.0f*d1-d2;
    float d3=ph[gc(k,j+3*dy,i+3*dx,nx,ny)]-ph[gc(k,j-3*dy,i-3*dx,nx,ny)];
    return 45.0f*d1-9.0f*d2+d3;
}

extern "C" __global__ void global_rhs_phi(
    int nx,int ny,int nz,int order,int phi_adv_z,float rdx,float rdy,float cfn,float cfn1,
    const float* c1,const float* c2,const float* fnm,const float* fnp,const float* rdnw,
    const float* ux,const float* vy,const float* ty,const float* muu,const float* muv,
    const float* u,const float* v,const float* ww,const float* ph,const float* phb,float* tend) {
    int q=blockIdx.x*blockDim.x+threadIdx.x,mp=nx*ny;
    if(q>=(nz+1)*mp)return;
    int k=q/mp,xy=q%mp,j=xy/nx,i=xy%nx,up=ny*(nx+1),vp=(ny+1)*nx;
    if(k==0)return;
    float value=tend[q];
    if(k<nz) {
        float w0=rdnw[k-1]*(ph[q]-ph[q-mp]+phb[q]-phb[q-mp]);
        float w1=rdnw[k]*(ph[q+mp]-ph[q]+phb[q+mp]-phb[q]);
        if(phi_adv_z==2)value=value-ww[q]*(fnm[k]*w1+fnp[k]*w0);
        else {
            w0=0.5f*(ww[q]+ww[q-mp])*rdnw[k-1]*(ph[q]-ph[q-mp]+phb[q]-phb[q-mp]);
            w1=0.5f*(ww[q+mp]+ww[q])*rdnw[k]*(ph[q+mp]-ph[q]+phb[q+mp]-phb[q]);
            value=value-(fnm[k]*w1+fnp[k]*w0);
        }
    }
    float scale=k==nz?0.5f:0.25f;
    float norm=order<=2?1.0f:order<=4?(1.0f/12.0f):(1.0f/60.0f);
    for(int axis=1;axis>=0;--axis) {
        int faces[2]={axis==0?j*(nx+1)+i:j*nx+i,
                      axis==0?j*(nx+1)+i+1:(j+1)*nx+i};
        float vel[2];
        for(int s=0;s<2;++s) {
            int f=faces[s],stride=axis==0?up:vp;
            const float* wind=axis==0?u:v;
            float windsum=k==nz?(cfn*wind[(k-1)*stride+f]+cfn1*wind[(k-2)*stride+f]):
                                (wind[k*stride+f]+wind[(k-1)*stride+f]);
            float mass=axis==0?muu[f]:muv[f],metric=axis==0?ux[f]:vy[f];
            vel[s]=(c1[k]*mass+c2[k])*windsum*metric;
        }
        float dx=axis==0?rdx:rdy;
        if(order<=2) {
            int di=axis==0,dj=axis==1;
            size_t a=gc(k,j+dj,i+di,nx,ny),b=gc(k,j-dj,i-di,nx,ny);
            float diff1=phb[a]-phb[q]+ph[a]-ph[q];
            float diff0=phb[q]-phb[b]+ph[q]-ph[b];
            value=value-(scale*dx/ty[xy])*(vel[1]*diff1+vel[0]*diff0);
        } else {
            // Native keeps the ph and base derivative terms in the same
            // left-associated expression (not two pre-rounded derivatives).
            float derivative;
            int di=axis==0,dj=axis==1;
            float d1=ph[gc(k,j+dj,i+di,nx,ny)]-ph[gc(k,j-dj,i-di,nx,ny)];
            float d2=ph[gc(k,j+2*dj,i+2*di,nx,ny)]-ph[gc(k,j-2*dj,i-2*di,nx,ny)];
            float b1=phb[gc(k,j+dj,i+di,nx,ny)]-phb[gc(k,j-dj,i-di,nx,ny)];
            float b2=phb[gc(k,j+2*dj,i+2*di,nx,ny)]-phb[gc(k,j-2*dj,i-2*di,nx,ny)];
            if(order<=4)derivative=8.0f*d1-d2+8.0f*b1-b2;
            else {
                float d3=ph[gc(k,j+3*dj,i+3*di,nx,ny)]-ph[gc(k,j-3*dj,i-3*di,nx,ny)];
                float b3=phb[gc(k,j+3*dj,i+3*di,nx,ny)]-phb[gc(k,j-3*dj,i-3*di,nx,ny)];
                derivative=45.0f*d1-9.0f*d2+d3+45.0f*b1-9.0f*b2+b3;
            }
            value=value-(scale*dx/ty[xy])*((vel[1]+vel[0])*norm*derivative);
        }
    }
    tend[q]=value;
}
