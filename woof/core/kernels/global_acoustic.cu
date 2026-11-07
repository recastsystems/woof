// WRF v4.6.1 module_small_step_em.F, public domain.
// Hydrostatic advance_uv and directional advance_mu_t. Input ghosts are
// explicit so native replays include the complete stencil read footprint.
static __device__ __forceinline__ size_t gh(int k,int j,int i,int nx,int ny) {
    return ((size_t)k*(ny+2)+j+1)*(nx+2)+i+1;
}

extern "C" __global__ void global_pack_one(
    int nx,int ny,int nz,const float* src,float* halo) {
    int q=blockIdx.x*blockDim.x+threadIdx.x,w=nx+2,h=ny+2;
    if(q>=nz*w*h) return;
    int i=q%w-1,j=(q/w)%h-1,k=q/(w*h);
    int ii=(i+nx)%nx,jj=max(0,min(ny-1,j));
    halo[q]=src[((size_t)k*ny+jj)*nx+ii];
}

extern "C" __global__ void global_advance_uv(
    int nx,int ny,int nz,float rdx,float rdy,float dt,float emdiv,
    const float* c1,const float* c2,
    const float* ux,const float* uy,const float* vx,const float* vy,const float* vxinv,
    const float* muu,const float* muv,const float* mudf,
    const float* ph,const float* alt,const float* p,const float* pb,const float* al,
    const float* ru_tend,const float* rv_tend,float* u,float* v) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    int up=ny*(nx+1),vp=(ny+1)*nx;
    if(q<nz*up) {
        int i=q%(nx+1),j=(q/(nx+1))%ny,k=q/up;
        int xy=j*(nx+1)+i;
        size_t a=gh(k,j,i,nx,ny),b=gh(k,j,i-1,nx,ny);
        size_t aa=gh(k+1,j,i,nx,ny),bb=gh(k+1,j,i-1,nx,ny);
        float dp=(ux[xy]/uy[xy])*0.5f*rdx*(c1[k]*muu[xy]+c2[k])*
            (((ph[aa]-ph[bb])+(ph[a]-ph[b]))
             +(alt[a]+alt[b])*(p[a]-p[b])+(al[a]+al[b])*(pb[a]-pb[b]));
        float md=(-emdiv*(1.0f/rdx)*(mudf[gh(0,j,i,nx,ny)]
                                           -mudf[gh(0,j,i-1,nx,ny)]))/uy[xy];
        float value=u[q]+dt*ru_tend[q];
        u[q]=value-dt*dp+c1[k]*md;
    }
    if(q<nz*vp) {
        int i=q%nx,j=(q/nx)%(ny+1),k=q/vp;
        if(j==0 || j==ny) { v[q]=0.0f; return; }
        int xy=j*nx+i;
        size_t a=gh(k,j,i,nx,ny),b=gh(k,j-1,i,nx,ny);
        size_t aa=gh(k+1,j,i,nx,ny),bb=gh(k+1,j-1,i,nx,ny);
        float dp=(vy[xy]/vx[xy])*0.5f*rdy*(c1[k]*muv[xy]+c2[k])*
            (((ph[aa]-ph[bb])+(ph[a]-ph[b]))
             +(alt[a]+alt[b])*(p[a]-p[b])+(al[a]+al[b])*(pb[a]-pb[b]));
        float md=(-emdiv*(1.0f/rdy)*(mudf[gh(0,j,i,nx,ny)]
                                           -mudf[gh(0,j-1,i,nx,ny)]))*vxinv[xy];
        float value=v[q]+dt*rv_tend[q];
        v[q]=value-dt*dp+c1[k]*md;
    }
}

extern "C" __global__ void global_advance_mass_theta(
    int nx,int ny,int nz,float rdx,float rdy,float dt,float epssm,
    const float* c1,const float* c2,const float* dnw,
    const float* fnm,const float* fnp,const float* rdnw,
    const float* tx,const float* ty,const float* uy,const float* vxinv,
    const float* muu,const float* muv,const float* mut,
    const float* u,const float* v,const float* u_ref,const float* v_ref,
    const float* theta_ref,const float* ww_ref,
    const float* theta_tend,const float* mu_tend,
    float* mu,float* muts,float* muave,float* mudf,
    float* ww,float* theta,float* theta_ave,float* divv) {
    int xy=blockIdx.x*blockDim.x+threadIdx.x,plane=nx*ny;
    if(xy>=plane) return;
    int j=xy/nx,i=xy%nx,up=ny*(nx+1),vp=(ny+1)*nx;
    int u0=j*(nx+1)+i,u1=u0+1,v0=j*nx+i,v1=v0+nx;
    float dmdt=0.0f;
    for(int k=0;k<nz;++k) {
        float dv=tx[xy]*ty[xy]*(
            rdy*((v[k*vp+v1]+(c1[k]*muv[v1]+c2[k])*v_ref[k*vp+v1]*vxinv[v1])
                -(v[k*vp+v0]+(c1[k]*muv[v0]+c2[k])*v_ref[k*vp+v0]*vxinv[v0]))
            +rdx*((u[k*up+u1]+(c1[k]*muu[u1]+c2[k])*u_ref[k*up+u1]/uy[u1])
                 -(u[k*up+u0]+(c1[k]*muu[u0]+c2[k])*u_ref[k*up+u0]/uy[u0])));
        divv[k*plane+xy]=dv;
        dmdt=dmdt+dnw[k]*dv;
    }
    float old=mu[xy];
    mu[xy]=mu[xy]+dt*(dmdt+mu_tend[xy]);
    mudf[xy]=dmdt+mu_tend[xy];
    muts[xy]=mut[xy]+mu[xy];
    muave[xy]=0.5f*((1.0f+epssm)*mu[xy]+(1.0f-epssm)*old);
    for(int k=1;k<nz;++k)
        ww[k*plane+xy]=ww[(k-1)*plane+xy]-dnw[k-1]*
            ((c1[k-1]*dmdt)+divv[(k-1)*plane+xy]+(c1[k-1]*mu_tend[xy]))/ty[xy];
    for(int k=0;k<nz;++k) ww[k*plane+xy]=ww[k*plane+xy]-ww_ref[k*plane+xy];
    for(int k=0;k<nz;++k) {
        int at=k*plane+xy;
        theta_ave[at]=theta[at];
        float value=theta[at]+ty[xy]*dt*theta_tend[at];
        float wd0=0.0f,wd1=0.0f;
        if(k>0) wd0=ww[at]*(fnm[k]*theta_ref[gh(k,j,i,nx,ny)]
                                  +fnp[k]*theta_ref[gh(k-1,j,i,nx,ny)]);
        if(k+1<nz) wd1=ww[at+plane]*(fnm[k+1]*theta_ref[gh(k+1,j,i,nx,ny)]
                                           +fnp[k+1]*theta_ref[gh(k,j,i,nx,ny)]);
        value=value-dt*ty[xy]*(tx[xy]*(
            0.5f*rdy*(v[k*vp+v1]*(theta_ref[gh(k,j+1,i,nx,ny)]+theta_ref[gh(k,j,i,nx,ny)])
                     -v[k*vp+v0]*(theta_ref[gh(k,j,i,nx,ny)]+theta_ref[gh(k,j-1,i,nx,ny)]))
            +0.5f*rdx*(u[k*up+u1]*(theta_ref[gh(k,j,i+1,nx,ny)]+theta_ref[gh(k,j,i,nx,ny)])
                      -u[k*up+u0]*(theta_ref[gh(k,j,i,nx,ny)]+theta_ref[gh(k,j,i-1,nx,ny)])))
            +rdnw[k]*(wd1-wd0));
        theta[at]=value;
    }
}
