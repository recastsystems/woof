// SMS-3DTKE's native field views, using the existing WRF metric helpers.
// smag2d.cu is included only in this module; its other compiled routes stay unchanged.
extern "C" __global__ void km5_geometry(const real* u,const real* v,const real* w,
    const real* php,const real* phb,const real* alt,const real* qv,
    const real* msft,const real* msfu,const real* msfv,
    const real* fnm,const real* fnp,const real* dn,const real* dnw,
    real rdx,real rdy,real dx,real dy,real cf1,real cf2,real cf3,int moist,
    const real* thp,const real* thb,int thb3d,const real* pressure,
    real* theta,real* rho,real* up,real* vp,real* rdza,real* rdzwa,
    real* zxa,real* zya,real* d33,real* d13,real* d23,real* p8w,real* t8w,
    int nz,int ny,int nx,int phb3d,int boundary_x,int boundary_y) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y,k=blockIdx.z;
    if(i>=nx||j>=ny||k>nz)return;
    WrfSmagGrid q=wrf_grid(u,v,w,php,phb,alt,qv,msft,msfu,msfv,
        fnm,fnp,dn,dnw,rdx,rdy,dx,dy,cf1,cf2,cf3,nz,ny,nx,phb3d,
        boundary_x,boundary_y,moist);
    SmagIndex h=I3(k,j,i,ny,nx);
    rdza[h]=wrf_rdz(q,k,j,i);
    d13[h]=wrf_defor13(q,k,j,i);d23[h]=wrf_defor23(q,k,j,i);
    zxa[I3(k,j,i,ny,nx+1)]=wrf_zx(q,k,j,i);
    zya[I3(k,j,i,ny+1,nx)]=wrf_zy(q,k,j,i);
    if(i==nx-1)zxa[I3(k,j,nx,ny,nx+1)]=wrf_zx(q,k,j,nx);
    if(j==ny-1)zya[I3(k,ny,i,ny+1,nx)]=wrf_zy(q,k,ny,i);
    if(k<nz) {
        theta[h]=wrf_theta_full(q,thp,thb,thb3d,k,j,i);
        // phy_prep uses vapor-loaded density, as does the shared diffusion.
        rho[h]=wrf_rho(q,k,j,i);
        up[h]=0.5f*(u[I3(k,j,i,ny,nx+1)]+u[I3(k,j,i+1,ny,nx+1)]);
        vp[h]=0.5f*(v[I3(k,j,i,ny+1,nx)]+v[I3(k,j+1,i,ny+1,nx)]);
        rdzwa[h]=wrf_rdzw(q,k,j,i);
        d33[h]=2.0f*(w[I3(k+1,j,i,ny,nx)]-w[h])*rdzwa[h];
    }
    if(k==0||k==nz) {
        int ka=k==0?0:nz-1,kb=k==0?1:nz-2;
        real z0=__fdiv_rn(wrf_phi(q,k,j,i),G);
        real z1=wrf_phy_z(q,ka,j,i),z2=wrf_phy_z(q,kb,j,i);
        real w1=(z0-z2)/(z1-z2),w2=1.0f-w1;
        real pa=pressure[I3(ka,j,i,ny,nx)],pb=pressure[I3(kb,j,i,ny,nx)];
        real ta=wrf_theta_full(q,thp,thb,thb3d,ka,j,i)*gfk_pow(__fdiv_rn(pa,P0),RCP);
        real tb=wrf_theta_full(q,thp,thb,thb3d,kb,j,i)*gfk_pow(__fdiv_rn(pb,P0),RCP);
        p8w[h]=k==0?w1*pa+w2*pb:gfk_exp(w1*gfk_log(pa)+w2*gfk_log(pb));
        t8w[h]=w1*ta+w2*tb;
    }
}
